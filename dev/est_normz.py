
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
    P = 10
    N_SPIX = 4
    N_PIX = 16

    # -- allocate data --
    attn = th.exp(th.ones((B,HD,N_SPIX,P,P),device=device)).double().requires_grad_(True)
    # sims = th.rand((B,N_SPIX,N_PIX),device=device).double().requires_grad_(True)
    sims = th.rand(1,device=device)*th.ones((B,N_SPIX,N_PIX),device=device).double()
    sims = sims.requires_grad_(True)
    nsamples = 100

    # -- check attn --
    fwd_fxn0 = lambda x: EffNormzFunction.apply(x,sims,nsamples)
    print(fwd_fxn0(attn))
    th.autograd.gradcheck(fwd_fxn0, attn, eps=1e-4,
                          atol=1e-2, nondet_tol=1e-7, raise_exception=True)

    # -- check sims --
    fwd_fxn1 = lambda y: EffNormzFunction.apply(attn,y,nsamples)
    th.autograd.gradcheck(fwd_fxn1, sims, eps=1e-4,
                          atol=1e-2, nondet_tol=1e-7, raise_exception=True)



if __name__ == "__main__":
    main()
