
# -- imports --
import math
import torch as th
from einops import rearrange
from superpixel_paper.est_attn_normz import EffNormzFunction
from stnls.testing.gradcheck import gradcheck_skipnan,gradcheck_skip_nan_unstable

# -- bench --
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt

def fwd_v2(attn,samples):
    # acc += attn[bi][spi][hi][pi][pk]*samples[bi][spi][pk][si];
    samples = samples[:,:,None]
    out = attn @ samples
    print(attn.shape,samples.shape,out.shape)
    return out

def main():

    # -- config --
    device = "cuda:0"
    B,HD = 6,1
    S = 1
    P = 25
    # P = 9
    N_PIX = 128*128
    N_SPIX = (128//S)*(128//S)
    # N_SPIX = 15
    th.manual_seed(123)

    # -- allocate data --
    attn = th.rand((B,N_SPIX,HD,P,P),device=device).double().requires_grad_(True)
    nsamples = 10

    # -- sampling --
    sims = th.rand((B,N_SPIX,P),device=device).double()
    sims = sims[...,None].expand(-1,-1,-1,nsamples)
    samples = th.bernoulli(sims)
    # print(sims.shape,samples.shape)
    # exit()

    # -- init bench --
    timer,memer = ExpTimer(),GpuMemer()


    # -- benchmark --
    attn0 = attn.clone().requires_grad_(True)
    fwd_fxn0 = lambda x: EffNormzFunction.apply(x,samples)
    with TimeIt(timer,"fwd"):
        with MemIt(memer,"fwd"):
            out = fwd_fxn0(attn0)
    attn0.retain_grad()
    loss = out.abs().mean()
    with TimeIt(timer,"bwd"):
        with MemIt(memer,"bwd"):
            loss.backward()

    # -- benchmark --
    attn0 = attn.clone().requires_grad_(True)
    fwd_fxn0 = lambda x: EffNormzFunction.apply(x,samples)
    with TimeIt(timer,"fwd_1"):
        with MemIt(memer,"fwd_1"):
            out = fwd_fxn0(attn0)
    attn0.retain_grad()
    loss = out.abs().mean()
    with TimeIt(timer,"bwd_1"):
        with MemIt(memer,"bwd_1"):
            loss.backward()

    # -- benchmark [v2] --
    attn1 = attn.clone().requires_grad_(True)
    with TimeIt(timer,"fwd_v2"):
        with MemIt(memer,"fwd_v2"):
            out2 = fwd_v2(attn1,samples)
    attn1.retain_grad()
    loss = out2.abs().mean()
    with TimeIt(timer,"bwd_v2"):
        with MemIt(memer,"bwd_v2"):
            loss.backward()

    # -- benchmark [v2] --
    attn1 = attn.clone().requires_grad_(True)
    with TimeIt(timer,"fwd_v2_1"):
        with MemIt(memer,"fwd_v2_1"):
            out2 = fwd_v2(attn1,samples)
    attn1.retain_grad()
    loss = out2.abs().mean()
    with TimeIt(timer,"bwd_v2_1"):
        with MemIt(memer,"bwd_v2_1"):
            loss.backward()


    print(timer)
    print(memer)

    # -- compare --
    out2 = fwd_fxn0(attn0)
    print(th.mean((out-out2)**2))
    return

    # -- check attn --
    # print(th.mean((fwd_fxn0(attn)-fwd_fxn0(attn))**2))
    th.autograd.gradcheck(fwd_fxn0, attn, eps=1e-5,
                          atol=1e-5, nondet_tol=1e-5, raise_exception=True)
    return



if __name__ == "__main__":
    main()
