
# -- imports --
import math
import torch as th
from einops import rearrange
from superpixel_paper.ssna import SoftSuperpixelNeighborhoodAttention

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
    B,HD = 1,1
    sp_token = 2
    H,W = 16,16
    sH,sW = (H-1)//sp_token,(W-1)//sp_token
    dim = 1
    kernel_size = 3
    th.manual_seed(123)

    # -- allocate data --
    img = th.rand((B,dim,H,W),device=device).double()
    sims = th.rand((B,H,W,sH,sW),device=device).double()

    # -- init layer --
    ssna = SoftSuperpixelNeighborhoodAttention(dim,HD,kernel_size,mask_labels=False)
    ssna = ssna.to(device).double()

    # -- init bench --
    timer,memer = ExpTimer(),GpuMemer()

    # -- benchmark --
    img0 = img.clone().requires_grad_(True)
    fwd_fxn0 = lambda x: ssna(x,sims)
    with TimeIt(timer,"fwd"):
        with MemIt(memer,"fwd"):
            out = fwd_fxn0(img0)
    img0.retain_grad()
    loss = out.abs().mean()
    with TimeIt(timer,"bwd"):
        with MemIt(memer,"bwd"):
            loss.backward()

    print(timer)
    print(memer)

    # -- check attn --
    img0 = img.clone().requires_grad_(True)
    fwd_fxn = lambda x: ssna(x,sims)
    th.autograd.gradcheck(fwd_fxn, img0, eps=1e-5,
                          atol=1e-5, nondet_tol=1e-5, raise_exception=True)

    sims = sims.clone().requires_grad_(True)
    fwd_fxn = lambda x: ssna(img,x)
    th.autograd.gradcheck(fwd_fxn, sims, eps=1e-5,
                          atol=1e-5, nondet_tol=1e-5, raise_exception=True)

    return


if __name__ == "__main__":
    main()
