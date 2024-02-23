
# -- pytorch --
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# -- basics --
import math
from einops import rearrange

# -- superpixel --
from superpixel_paper.est_attn_normz import EffNormzFunction
from superpixel_paper.sr_models.pair_wise_distance import PairwiseDistFunction
from natten import NeighborhoodAttention2D


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        dim (int): Channel number of intermediate features.
        reduction (int): Channel reduction factor. Default: 16.
    """

    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class FFN(nn.Module):
    """Feed Forward Network.
    Args:
        dim (int): Base channels.
        hidden_dim (int): Channels of hidden mlp.
    """

    def __init__(self, dim, hidden_dim, out_dim, norm_layer=LayerNorm2d):
        super().__init__()
        self.norm = norm_layer(dim)

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, dim, 1, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, bias=True),
            nn.Sigmoid()
        )

        self.fc1 = nn.Conv2d(dim * 2, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = torch.cat([self.ca(x) * x, self.pa(x) * x], dim=1)
        return self.fc2(self.act(self.fc1(x)))


class VanillaAttention(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
    """

    def __init__(self, dim, heads, qk_dim):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        """Forward function.
        If 'y' is None, it performs self-attention; Otherwise it performs cross-attention.
        Args:
            x (Tensor): Input feature.
            y (Tensor): Support feature.
        Returns:
            out(Tensor): Output feature.
        """
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out


class LocalTokenAttention(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
    """

    def __init__(self, dim, heads, qk_dim, stoken_size):
        super().__init__()

        self.stoken_size = stoken_size
        self.norm = nn.LayerNorm(dim)
        self.attn = VanillaAttention(dim, heads, qk_dim)

    def local_partition(self, x, h_step, w_step, dh, dw):
        b, c, h, w = x.size()
        local_x = []
        for i in range(0, h + h_step - dh, h_step):
            top = i
            down = i + dh
            if down > h:
                top = h - dh
                down = h
            for j in range(0, w + w_step - dw, w_step):
                left = j
                right = j + dw
                if right > w:
                    left = w - dw
                    right = w
                local_x.append(x[:, :, top:down, left:right])
        local_x = torch.stack(local_x, dim=0) 
        local_x = local_x.permute(1, 0, 2, 3, 4).contiguous()
        return local_x

    def local_reverse(self, local_x, x, h_step, w_step, dh, dw):
        b, c, h, w = x.size()
        x_output = torch.zeros_like(x)
        count = torch.zeros((b,h,w), device=x.device)

        index = 0
        for i in range(0, h + h_step - dh, h_step):
            top = i
            down = i + dh
            if down > h:
                top = h - dh
                down = h
            for j in range(0, w + w_step - dw, w_step):
                left = j
                right = j + dw
                if right > w:
                    left = w - dw
                    right = w
                x_output[:, :, top:down, left:right] += local_x[:, index]
                count[:, top:down, left:right] += 1
                index += 1
        x_output = x_output / count.unsqueeze(1)
        return x_output

    def forward(self, x):
        """Forward function.
        If 'y' is None, it performs self-attention; Otherwise it performs cross-attention.
        Args:
            x (Tensor): Input feature.
            y (Tensor): Support feature.
        Returns:
            out(Tensor): Output feature.
        """

        B, C, H, W = x.shape
        dh, dw = self.stoken_size[0], self.stoken_size[1]

        # import pdb; pdb.set_trace()
        local_x = self.local_partition(x, dh-2, dw-2, dh, dw)
        _, n, _, dh, dw = local_x.shape
        local_x = rearrange(local_x, 'b n c dh dw -> (b n) (dh dw) c')

        local_x = self.attn(self.norm(local_x))
        local_x = rearrange(local_x, '(b n) (dh dw) c  -> b n c dh dw', n=n, dw=dw)
        x = self.local_reverse(local_x, x, dh-2, dw-2, dh, dw)

        return x


