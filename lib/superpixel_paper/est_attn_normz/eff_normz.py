
import torch
import torch as th
import eff_normz_cuda


class EffNormzFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, attn, samples):

        # -- unpack --
        B,K,HD,P0,_ = attn.shape
        attn = attn.contiguous()
        # samples.shape = (B,K,P0,NSAMPLES)

        # -- allocate --
        normz = th.zeros_like(attn)

        # -- forward --
        eff_normz_cuda.eff_forward(attn,samples,normz)

        # -- save --
        ctx.save_for_backward(samples,attn)
        return normz

    @staticmethod
    def backward(ctx, normz_grad):

        # -- unpack --
        samples,attn = ctx.saved_tensors

        # -- allocate --
        attn_grad = th.zeros_like(attn)#normz_grad)

        # -- backward --
        eff_normz_cuda.eff_backward_og(attn_grad,normz_grad.contiguous(),attn,samples)
        return attn_grad, None

    @staticmethod
    def sample(sims, nsamples):
        # -- sampling --
        sims = sims[...,None].expand(-1,-1,-1,nsamples)
        samples = th.bernoulli(sims)
        return samples

