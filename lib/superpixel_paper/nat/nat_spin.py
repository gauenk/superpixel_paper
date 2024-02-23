"""
Neighborhood Attention 2D PyTorch Module

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
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
        dim,
        num_heads,
        kernel_size,
        dilation=1,
        bias=True,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        learn_attn_scale=False,
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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        _bool = (learn_attn_scale is None) or (learn_attn_scale is False)
        self.learn_attn_scale = not(_bool)
        if (learn_attn_scale is None) or (learn_attn_scale is False):
            self.attn_scale_net = nn.Identity()
        else:
            self.attn_scale_net = learn_attn_scale

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, H, W, 3, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # -- rescaling --
        if self.learn_attn_scale:
            scale = self.attn_scale_net(rearrange(x,'b h w c -> b c h w'))
            scale = rearrange(scale,'b 1 h w -> b 1 h w 1')
            q = scale * q
        else:
            q = self.scale * q

        attn = natten2dqkrpb(q, k, self.rpb, self.kernel_size, self.dilation)
        # attn_0 = attn.clone()
        # print("attn.shape: ",attn.shape)
        # print(self.scale)
        # exit()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = natten2dav(attn, v, self.dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rel_pos_bias={self.rpb is not None}"
        )




class NeighAttnMat(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
        self,
        dim,
        num_heads,
        kernel_size,
        dilation=1,
        bias=True,
        qk_bias=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.qk = nn.Linear(dim, dim * 2, bias=qk_bias)
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qk = (
            self.qk(x)
            .reshape(B, H, W, 2, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k = qk[0], qk[1]
        attn = natten2dqkrpb(q, k, self.rpb, self.kernel_size, self.dilation)
        return attn

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rel_pos_bias={self.rpb is not None}"
        )


class NeighAttnAgg(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        kernel_size,
        dilation=1,
        v_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()

        # -- for padding --
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.v = nn.Linear(dim, dim * 1, bias=v_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn):
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        v = (self.v(x)
            .reshape(B, H, W, 1, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 1, 2, 5))[0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = natten2dav(attn, v, self.dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
        )

def natten_padding(x,kernel_size):
    window_size = kernel_size*kernel_size
    B, Hp, Wp, C = x.shape
    H, W = int(Hp), int(Wp)
    pad_l = pad_t = pad_r = pad_b = 0
    if H < window_size or W < window_size:
        pad_l = pad_t = 0
        pad_r = max(0, window_size - W)
        pad_b = max(0, window_size - H)
        x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, H, W, _ = x.shape
    pad_info = {"Hp":Hp,"Wp":Wp,"pad_r":pad_r,"pad_b":pad_b}
    return x,pad_info

def natten_remove_padding(x,pad_info):
    Hp,Wp = pad_info["Hp"],pad_info["Wp"]
    pad_r,pad_b = pad_info["pad_r"],pad_info["pad_b"]
    if pad_r or pad_b:
        x = x[:, :Hp, :Wp, :]
    return x
