
# -- imports --
import math
import torch as th
from einops import rearrange
from superpixel_paper.est_attn_normz import EffNormzFunction
from stnls.testing.gradcheck import gradcheck_skipnan,gradcheck_skip_nan_unstable

def main():

    # -- config --
    device = "cuda:0"
    B,HD = 1,1
    P = 2
    N_SPIX = 2
    N_PIX = 2
    th.manual_seed(123)

    # -- allocate data --
    attn = th.exp(th.ones((B,HD,N_SPIX,P,P),device=device)).double().requires_grad_(True)
    # sims = th.rand((B,N_SPIX,N_PIX),device=device).double().requires_grad_(True)
    sims = 0.5*th.ones((B,N_SPIX,N_PIX),device=device).double()
    sims = sims.requires_grad_(True)
    # nsamples = 500000
    nsamples = 10

    # -- sampling --
    # sims.shape = B,K,nPixels
    sims = sims[...,None].expand(-1,-1,-1,nsamples)
    samples = th.bernoulli(sims)

    # -- check attn --
    fwd_fxn0 = lambda x: EffNormzFunction.apply(x,samples)
    # fwd_fxn0 = lambda x: EffNormzFunction.apply(x,sims,nsamples)
    print(th.mean((fwd_fxn0(attn)-fwd_fxn0(attn))**2))
    th.autograd.gradcheck(fwd_fxn0, attn, eps=1e-2,
                          atol=1e-2, nondet_tol=1e-5, raise_exception=True)

    # -- check sims --
    # fwd_fxn1 = lambda y: EffNormzFunction.apply(attn,y,nsamples)
    # th.autograd.gradcheck(fwd_fxn1, sims, eps=1e-4,
    #                       atol=1e-2, nondet_tol=1e-7, raise_exception=True)



if __name__ == "__main__":
    main()
