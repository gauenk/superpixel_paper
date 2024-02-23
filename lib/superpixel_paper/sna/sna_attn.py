
import torch
import torch as th

import torch
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_
from einops import rearrange


import torch
from torch.autograd import Function
import superpixel_cuda
# from torch.cuda.amp import custom_bwd, custom_fwd

from natten.functional import natten2dav, natten2dqkrpb

class NeighSuperpixelAttn(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(self,
                 dim,
                 num_heads,
                 kernel_size,
                 dilation=1,
                 qk_bias=True,
                 use_weights=True,
                 qk_layer=None,
                 qk_scale=None,
                 learn_attn_scale=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.qk_scale = qk_scale or self.head_dim**-0.5
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
        # if bias:
        #     self.rpb = nn.Parameter(
        #         torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
        #     )
        #     trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        # else:
        #     self.register_parameter("rpb", None)


        # -- scaling q --
        _bool = (learn_attn_scale is None) or (learn_attn_scale is False)
        self.learn_attn_scale = not(_bool)
        if (learn_attn_scale is None) or (learn_attn_scale is False):
            self.attn_scale_net = nn.Identity()
        else:
            self.attn_scale_net = learn_attn_scale

        # -- viz --
        self.q_shell = nn.Identity()
        self.k_shell = nn.Identity()
        self.imgSp_shell_attn = nn.Identity()
        self.attn_shell = nn.Identity()

    def forward(self, x, imgSp):
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

        # -- rescale --
        if self.learn_attn_scale:
            scale = self.attn_scale_net(rearrange(x,'b h w c -> b c h w'))
            # print(scale.shape)
            scale = rearrange(scale,'b 1 h w -> b 1 h w 1')
            # print("[1]: ",scale.shape,q.shape)
            # print(scale)
            q = scale * q
        else:
            q = self.qk_scale * q

        q = self.q_shell(q)
        k = self.k_shell(k)
        imgSp = self.imgSp_shell_attn(imgSp)
        # attn = natten2dqkrpb(q, k, None, self.kernel_size, 1)
        attn = NeighSuperpixelAttnFunction.apply(q,k,imgSp,self.kernel_size)
        attn = self.attn_shell(attn)
        return attn

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
        )


class NeighSuperpixelAttnFunction(Function):

    @staticmethod
    def forward(ctx, queries, keys, imgSp, kernel_size):

        # -- aggregate --
        dilation = 1
        assert dilation == 1
        queries = queries.contiguous()
        keys = keys.contiguous()

        B,HD,H,W,F = queries.shape
        # attn = th.zeros((B,HD,H,W,kernel_size**2)).to(queries.device)
        attn = -th.inf * th.ones((B,HD,H,W,kernel_size**2),
                                 dtype=queries.dtype).to(queries.device)
        superpixel_cuda.sna_attn_forward(attn, queries, keys, imgSp)
        ctx.save_for_backward(queries, keys, imgSp, attn)
        ctx.dilation = dilation
        # print("here: ",attn.min().item(),attn.max().item())
        # print(imgSp)
        # if th.any(th.isinf(attn)).item():
        #     print("hi.")
        #     exit()
        # print("yo.")
        # exit()
        #     print(th.any(th.isinf(attn)).item())
        #     print(th.where(th.isinf(attn)))
        #     print(queries.shape,keys.shape,attn.shape,kernel_size)
        #     exit()

        return attn

    @staticmethod
    def backward(ctx, d_attn):

        d_queries = th.zeros_like(ctx.saved_variables[0])
        d_keys = th.zeros_like(ctx.saved_variables[1])
        d_imgSp = th.zeros_like(ctx.saved_variables[2]).type(d_queries.dtype)
        superpixel_cuda.sna_attn_backward(
            d_queries,d_keys,d_imgSp,d_attn,
            ctx.saved_variables[0],
            ctx.saved_variables[1],
            ctx.saved_variables[2]
            # ctx.saved_variables[3]
        )
        return d_queries, d_keys, d_imgSp, None

