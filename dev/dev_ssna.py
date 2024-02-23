
# -- imports --
import math
import torch as th
from einops import rearrange
from superpixel_paper.ssna import SoftSuperpixelNeighborhoodAttention

# -- bench --
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt

def run_ssna_agg(ssna,img,sims,attn):
    img = img.permute(0,2,3,1) # b f h w -> b h w f
    H,W = img.shape[1],img.shape[2]
    sH,sW = sims.shape[-2:]
    sinds = ssna.get_indices(H,W,sH,sW,sims.device)
    out = ssna.nat_agg(img,attn,sims,sinds)
    out = out.permute(0,3,1,2).clone() # b h w f -> b f h w
    return out

def run_ssna_attn(ssna,img,sims):
    # -- reshape --
    img = img.permute(0,2,3,1) # b f h w -> b h w f
    # -- indices for access --
    device = img.device
    B,H,W,F = img.shape
    # H,W = img.shape[1],img.shape[2]
    sH,sW = sims.shape[-2:]
    sinds = ssna.get_indices(H,W,sH,sW,sims.device)
    # -- attn map --
    # print(img.shape,sims.shape,sinds.shape)
    # attn = ssna.nat_attn(img,sims,sinds)
    labels = th.zeros((B,H,W),device=device,dtype=th.long)
    attn = ssna.nat_attn(img,labels)
    attn = ssna.attn_rw(attn,sims,sinds)
    return attn

def main():

    # -- config --
    device = "cuda:0"
    B,HD = 2,1
    sp_token = 2
    H,W = 16,16
    sH,sW = H//sp_token,W//sp_token
    dim = 1
    kernel_size = 3
    th.manual_seed(123)

    # -- allocate data --
    img = th.rand((B,dim,H,W),device=device).double()
    sims = th.rand((B,H,W,sH,sW),device=device).double()
    # sims = th.ones_like(sims)/9.

    # -- init layer --
    ssna = SoftSuperpixelNeighborhoodAttention(dim,HD,kernel_size,mask_labels=False)
    ssna = ssna.to(device).double()

    # -- init bench --
    timer,memer = ExpTimer(),GpuMemer()

    # -- benchmark --
    img0 = img.clone().requires_grad_(True)
    fwd_fxn0 = lambda x: ssna(x,sims)
    out = fwd_fxn0(img0)
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

    # return

    # -- check attn --
    # img0 = img.clone().requires_grad_(True)
    # fwd_fxn = lambda x: run_ssna_attn(ssna,x,sims)
    # th.autograd.gradcheck(fwd_fxn, img0, eps=1e-5,
    #                       atol=1e-5, nondet_tol=1e-5, raise_exception=True)
    # sims0 = sims.clone().requires_grad_(True)
    # fwd_fxn = lambda x: run_ssna_attn(ssna,img,x)
    # th.autograd.gradcheck(fwd_fxn, sims0, eps=1e-5,
    #                       atol=1e-5, nondet_tol=1e-5, raise_exception=True)
    # return


    # -- check agg --
    # attn = run_ssna_attn(ssna,img,sims).clone().detach()
    # img0 = img.clone().requires_grad_(True)
    # fwd_fxn = lambda x: run_ssna_agg(ssna,x,sims,attn)
    # th.autograd.gradcheck(fwd_fxn, img0, eps=1e-5,
    #                       atol=1e-5, nondet_tol=1e-5, raise_exception=True)
    # sims0 = sims.clone().requires_grad_(True)
    # fwd_fxn = lambda x: run_ssna_agg(ssna,img,x,attn)
    # th.autograd.gradcheck(fwd_fxn, sims0, eps=1e-5,
    #                       atol=1e-5, nondet_tol=1e-5, raise_exception=True)
    # attn0 = attn.clone().requires_grad_(True)
    # fwd_fxn = lambda x: run_ssna_agg(ssna,img,sims,x)
    # th.autograd.gradcheck(fwd_fxn, attn0, eps=1e-5,
    #                       atol=1e-5, nondet_tol=1e-5, raise_exception=True)
    # return


    # -- check attn --
    img0 = img.clone().requires_grad_(True)
    fwd_fxn = lambda x: ssna(x,sims)
    th.autograd.gradcheck(fwd_fxn, img0, eps=1e-5,
                          atol=1e-5, nondet_tol=1e-5, raise_exception=True)

    sims0 = sims.clone().requires_grad_(True)
    fwd_fxn = lambda x: ssna(img,x)
    th.autograd.gradcheck(fwd_fxn, sims0, eps=1e-5,
                          atol=1e-5, nondet_tol=1e-5, raise_exception=True)

    return


if __name__ == "__main__":
    main()
