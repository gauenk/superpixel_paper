
import torch
import torch as th

import torch
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_


import torch
from torch.autograd import Function
import superpixel_cuda
# from torch.cuda.amp import custom_bwd, custom_fwd

from natten.functional import natten2dav, natten2dqkrpb


class NeighSuperpixelAgg(nn.Module):
    """
    1D QK+RPB autograd function
    Computes neighborhood attention weights given queries and keys,
    and adds relative positional biases.
    This calls the `QKRPB` kernel.
    """


    def __init__(self,
                 dim,
                 num_heads,
                 kernel_size,
                 dilation=1,
                 v_bias=True,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 use_proj=True,
                 use_weights=True,
                 v_layer=None,proj_layer=None):
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
        self.use_weights = use_weights

        if self.use_weights:
            if v_layer is None:
                self.v = nn.Linear(dim, dim * 1, bias=v_bias)
            else:
                self.v = v_layer
        else:
            self.v = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)

        use_proj = True
        self.use_proj = use_proj
        if self.use_proj:
            if proj_layer is None:
                self.proj = nn.Linear(dim, dim)
            else:
                self.proj = proj_layer
            self.proj_drop = nn.Dropout(proj_drop)
        else:
            self.proj = nn.Identity()
            self.proj_drop = nn.Identity()

        # -- viz --
        self.v_shell = nn.Identity()

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
        attn = self.attn_drop(attn)
        v = self.v_shell(v)

        x = NeighSuperpixelAggFunction.apply(v,attn)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]
        x = self.proj_drop(self.proj(x))
        return x

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
        )


class NeighSuperpixelAggFunction(Function):

    @staticmethod
    def forward(ctx, values, attn):

        # -- aggregate --
        dilation = 1
        assert dilation == 1
        attn = attn.contiguous()
        values = values.contiguous()
        out = th.zeros_like(values)
        superpixel_cuda.sna_agg_forward(out, attn, values)
        ctx.save_for_backward(attn, values, out)
        ctx.dilation = dilation

        return out

    @staticmethod
    def backward(ctx, grad_imgOut):

        d_attn = th.zeros_like(ctx.saved_variables[0])
        d_imgV = th.zeros_like(ctx.saved_variables[1])

        superpixel_cuda.sna_agg_backward(
            d_attn,d_imgV,grad_imgOut,
            ctx.saved_variables[0],
            ctx.saved_variables[1],
            ctx.saved_variables[2],
        )
        return d_imgV, d_attn


# def nsa_aggregate(attn,value,weights,dilation):
#     return NeighSuperpixelAgg.apply(attn,value,weights,dilation)
