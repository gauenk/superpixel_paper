
import torch
import torch as th

import torch
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_


import torch
from torch.autograd import Function
import superpixel_cuda

from einops import rearrange
from .ssna_gather_sims import SsnaGatherSims

class AttnReweight(nn.Module):
    """

    Input: QK^T
    Output P(L_j=s)exp(QK^T)/[\sum_{j\in N(s)} P(L_j=s)exp(QK^T)]


    using "log(P(L_i=s))" gives us "nan" which we could ignore, but I just
    don't like having "nan"s in the pathway of my code.

    I'd rather do this without any "log P(L_i=s)" giving rows of nans.

    """

    def __init__(self):
        super().__init__()

    def forward(self, attn, sims, sinds):

        # -- P(L_j = s) --
        eps = 1e-10
        c = th.max(attn,dim=-1,keepdim=True).values
        attn = th.exp(attn-c)
        attn = AttnReweightFunction.apply(attn,sims,sinds)
        attn = attn / (eps+th.sum(attn,-1,keepdim=True))

        # -- reweight with P(L_i = s) --
        gather_sims = SsnaGatherSims()
        pi = gather_sims(sims,sinds)
        pi = rearrange(pi,'b h w ni -> b 1 ni h w 1')
        # print(pi.shape,attn.shape)
        # pi = pi/pi.sum(2,keepdim=True)
        attn = th.sum(pi * attn,2).contiguous()
        # print("[v0] attn.shape: ",attn.shape)
        # print(attn[0,0,0,0])
        # print(attn[0,0,1,1])

        return attn

    def extra_repr(self) -> str:
        return (f"attn reweight")

class AttnReweightFunction(Function):

    @staticmethod
    def forward(ctx, attn_in, sims, sinds):
        """

        Computes: P(Li = s) exp( d(qi,kj) )

        """
        NSP = 9
        dtype = attn_in.dtype
        device = attn_in.device
        B,HD,H,W,K = attn_in.shape
        attn_out = th.zeros((B,HD,NSP,H,W,K),device=device,dtype=dtype)
        superpixel_cuda.ssna_reweight_forward(attn_out, attn_in, sims, sinds)
        ctx.save_for_backward(attn_out, attn_in, sims, sinds)
        return attn_out

    @staticmethod
    def backward(ctx, d_attn_out):
        d_attn_out = d_attn_out.contiguous()
        d_attn_in = th.zeros_like(ctx.saved_variables[1])
        d_sims = th.zeros_like(ctx.saved_variables[2])
        superpixel_cuda.ssna_reweight_backward(
            d_attn_in,d_sims,d_attn_out,
            ctx.saved_variables[0],
            ctx.saved_variables[1],
            ctx.saved_variables[2],
            ctx.saved_variables[3],
        )

        return d_attn_in,d_sims,None

