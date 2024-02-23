
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

class SoftNeighSuperpixelAttn(nn.Module):
    """
    Soft Neighborhood Attention 2D Module
    """

    def __init__(
            self,
            dim,
            num_heads,
            kernel_size,
            dilation=1,
            qk_bias=True,
            use_weights=True,
            qk_layer=None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.use_weights = use_weights
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

        self.use_weights = use_weights
        if self.use_weights:
            if qk_layer is None:
                self.qk = nn.Linear(dim, dim * 2, bias=qk_bias)
            else:
                self.qk = qk_layer
            # self.qk = nn.Linear(dim, dim, bias=qk_bias)
        else:
            self.qk = nn.Identity()
        # self.qk = nn.Linear(dim, dim, bias=qk_bias)
        # self.qk = nn.Linear(dim, dim * 2, bias=qk_bias)
        # if bias:
        #     self.rpb = nn.Parameter(
        #         torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
        #     )
        #     trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        # else:
        #     self.register_parameter("rpb", None)

        # -- viz --
        self.q_shell = nn.Identity()
        self.k_shell = nn.Identity()
        self.sims_shell_attn = nn.Identity()
        self.attn_shell = nn.Identity()

    def forward(self, x, sims, sinds):
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape

        qk_num = 2 if self.use_weights else 1
        qk = (self.qk(x)
              .reshape(B, H, W, qk_num, self.num_heads, self.head_dim)
              .permute(3, 0, 4, 1, 2, 5))
        q, k = qk[0], qk[-1]

        q = self.q_shell(q)
        k = self.k_shell(k)
        sims = self.sims_shell_attn(sims)
        # attn = natten2dqkrpb(q, k, None, self.kernel_size, 1)
        attn = SoftNeighSuperpixelAttnFunction.apply(q,k,sims,sinds,self.kernel_size)
        attn = self.attn_shell(attn)
        return attn

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
        )


class SoftNeighSuperpixelAttnFunction(Function):

    @staticmethod
    def forward(ctx, queries, keys, sims, sinds, kernel_size):

        # -- aggregate --
        dilation = 1
        assert dilation == 1
        queries = queries.contiguous()
        keys = keys.contiguous()
        B,HD,H,W,F = keys.shape
        attn = -th.inf * th.ones((B,HD,9,H,W,kernel_size**2),
                                 dtype=queries.dtype).to(queries.device)
        superpixel_cuda.ssna_attn_forward(attn, queries, keys, sims, sinds)
        ctx.save_for_backward(queries, keys, attn, sims, sinds)
        ctx.dilation = dilation

        return attn

    @staticmethod
    def backward(ctx, d_attn):


        # -- allocate --
        d_queries = th.zeros_like(ctx.saved_variables[0])
        d_keys = th.zeros_like(ctx.saved_variables[1])
        d_sims = th.zeros_like(ctx.saved_variables[3])

        # -- exec --
        superpixel_cuda.ssna_attn_backward(
            d_queries,d_keys,d_sims,d_attn,
            ctx.saved_variables[0],
            ctx.saved_variables[1],
            ctx.saved_variables[2],
            ctx.saved_variables[3],
            ctx.saved_variables[4],
        )

        # tmp = th.stack([d_queries[0,0,:2,:2,0],d_keys[0,0,:2,:2,0]])
        # if th.any(tmp.abs()>0):
        #     tmp2 = d_queries[0,0,:2,:2,0]+d_keys[0,0,:2,:2,0]
        #     print(tmp)
        #     print(tmp2)
        #     print("-"*20)
        return d_queries, d_keys, d_sims, None, None

