import torch
import torch as th
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_
from einops import rearrange

from natten.functional import natten2dav, natten2dqkrpb

class NeighborhoodAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
        self,
        conv_q,
        conv_k,
        conv_v,
        proj,
        dim,
        num_heads,
        kernel_size,
        dilation=1,
        bias=False,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        self.conv_q = conv_q
        self.conv_k = conv_k
        self.conv_v = conv_v
        self.proj = proj
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = rearrange(x,'b c h w -> b h w c')
        # print("[0] x.shape: ",x.shape)
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        nheads = self.num_heads
        x = rearrange(x,'b h w c -> b c h w')
        q = rearrange(self.conv_q(x),'b (hd c) h w -> b hd h w c',hd=nheads)
        k = rearrange(self.conv_k(x),'b (hd c) h w -> b hd h w c',hd=nheads)
        v = rearrange(self.conv_v(x),'b (hd c) h w -> b hd h w c',hd=nheads)
        q = q * self.scale
        # print(q.shape,k.shape,v.shape)
        attn = natten2dqkrpb(q, k, self.rpb, self.kernel_size, self.dilation)
        # print(attns.shape)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = natten2dav(attn, v, self.dilation)
        # print("[0] x.shape: ",x.shape)
        x = rearrange(x,'b hd h w f -> b h w (hd f)')
        # print("[1] x.shape: ",x.shape)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]
        x = self.proj_drop(self.proj(x))
        x = rearrange(x,'b h w c -> b c h w')
        return x

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rel_pos_bias={self.rpb is not None}"
        )


