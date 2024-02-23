
import torch
import torch as th

import torch
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_

import torch
from torch.autograd import Function
import superpixel_cuda

class AttnReweight(nn.Module):
    """

    Input: QK^T
    Output \sum_{s}P(L_i=s) [P(L_j=s)exp(QK^T)]/(\sum_j[P(L_j=s)exp(QK^T)])


    using "log(P(L_i=s))" gives us "nan" which we could ignore, but I just
    don't like having "nan"s in the pathway of my code.

    I'd rather do this without any "log P(L_i=s)" giving rows of nans.

    """

    def __init__(self):
        super().__init__()

    def forward(self, attn, sims, sinds):
        eps = 1e-10
        c = attn.max().item()
        attn_rw = th.exp(attn-c)
        attn_rw = AttnReweightFunction.apply(attn_rw,sims,sinds)
        attn = attn_rw / (eps+th.sum(attn_rw,-1,keepdim=True))

        # -- part 2 --
        # attn = th.exp(attn[:,:,None] - c - th.log(eps+th.sum(attn_rw,-1)))
        # attn = AttnReduceFunction.apply(attn,sims,sinds)

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

