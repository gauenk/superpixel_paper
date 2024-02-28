
# -- imports --
import math
import torch as th
from einops import rearrange
from superpixel_paper.ssna.ssna import get_indices
from superpixel_paper.ssna.ssna_gather_sims import SsnaGatherSims
from superpixel_paper.ssna import SoftSuperpixelNeighborhoodAttention
from superpixel_paper.ssna.ssna_v2 import SoftSuperpixelNeighborhoodAttention_v2
from superpixel_paper.sna.sna import SuperpixelNeighborhoodAttention
from superpixel_paper.models.sp_modules import sparse_to_full
from superpixel_paper.nat.nat_spin import NeighborhoodAttention2D


# -- bench --
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt

def sample_sims(B,H,W,sH,sW,S,device):
    sims = th.rand((B,H,W,sH,sW),device=device).double()
    # sims = th.ones((B,H,W,sH,sW),device=device).double()
    # sims /= sims.sum((-1,-2),keepdim=True)
    gather_sims = SsnaGatherSims()
    sinds = get_indices(H,W,sH,sW,device)
    pi = gather_sims(sims,sinds)
    # print("pi.shape: ",pi.shape)
    # exit()
    pi = rearrange(pi,'b h w ni -> b ni h w')
    pi = pi/pi.sum(1,keepdim=True)
    sims = sparse_to_full(pi,S)
    sims = rearrange(sims,'b (sh sw) (h w) -> b h w sh sw',h=H,sh=sH)
    sims = sims.contiguous()

    # -- check --
    pi0 = gather_sims(sims,sinds)
    pi0 = rearrange(pi0,'b h w ni -> b ni h w')
    # print(pi0[0,:,0,0])
    # print(pi[0,:,0,0])
    # print("-"*10)
    # print(pi0[0,:,1,1])
    # print(pi[0,:,1,1])
    # print("-"*10)
    delta = th.mean((pi - pi0)**2)
    print("[delta]: ",delta)
    assert delta.item()<1e-3
    return sims


def main():

    # -- config --
    device = "cuda:0"
    B,HD = 2,1
    H,W = 16,16
    sp_token = 2
    kernel_size = 3
    # H,W = 128,128
    # H,W = 256,256
    # sp_token = 14
    # kernel_size = 9
    sH,sW = H//sp_token,W//sp_token
    dim = 3
    scale = 10.
    th.manual_seed(123)

    # -- allocate data --
    img = th.rand((B,dim,H,W),device=device).double()
    sims = sample_sims(B,H,W,sH,sW,sp_token,device)
    # print(sims.shape)
    # exit()
    # print(H,W,sH,sW,sims.shape)

    img[...] = 1.
    # # img[:,1] = 0.5
    # # img[:,2] = 0.8
    img[:,:,1,1] = 10.

    # -- init layer --
    nat = NeighborhoodAttention2D(dim=dim, kernel_size=kernel_size,
                                  num_heads=HD, bias=False, qkv_bias=False,
                                  qk_scale=scale, use_proj=False)
    sna = SuperpixelNeighborhoodAttention(dim,HD,kernel_size,qk_scale=scale,
                                          mask_labels=False,use_proj=False)
    ssna = SoftSuperpixelNeighborhoodAttention(dim,HD,kernel_size,qk_scale=scale,
                                               mask_labels=False,use_proj=False)
    ssna_v2 = SoftSuperpixelNeighborhoodAttention_v2(dim,HD,kernel_size,qk_scale=scale,
                                                     mask_labels=False,use_proj=False)
    nat = nat.to(device).double()
    sna = sna.to(device).double()
    ssna = ssna.to(device).double()
    ssna_v2 = ssna_v2.to(device).double()

    # -- set equal weights --
    mat = ssna.nat_agg.v.weight.data
    mat = th.eye(dim).to(device).double()
    nat.v.weight.data = mat
    sna.nat_agg.v.weight.data = mat
    ssna.nat_agg.v.weight.data = mat
    ssna_v2.nat_agg.v.weight.data = mat
    print(ssna.nat_attn.qk.weight.data.shape)
    mat = th.zeros(2*dim,dim).to(device)
    mat[:dim,:] = th.eye(dim).to(device)
    mat[-dim:,:] = th.eye(dim).to(device)
    nat.qk.weight.data = mat.double()
    sna.nat_attn.qk.weight.data = mat.double()
    ssna.nat_attn.qk.weight.data = mat.double()
    ssna_v2.nat_attn.qk.weight.data = mat.double()

    # -- init bench --
    timer,memer = ExpTimer(),GpuMemer()

    # -- benchmark [v2] --
    img1 = img.clone().requires_grad_(True)
    sims1 = sims.clone().requires_grad_(True)
    fwd_fxn1 = lambda x,s: ssna_v2(x,s)
    out1 = fwd_fxn1(img1,sims1)
    with TimeIt(timer,"fwd_v2"):
        with MemIt(memer,"fwd_v2"):
            out1 = fwd_fxn1(img1,sims1)
    img1.retain_grad()
    sims1.retain_grad()
    loss = out1.abs().mean()
    with TimeIt(timer,"bwd_v2"):
        with MemIt(memer,"bwd_v2"):
            loss.backward()

    # -- benchmark --
    img0 = img.clone().requires_grad_(True)
    sims0 = sims.clone().requires_grad_(True)
    fwd_fxn0 = lambda x,s: ssna(x,s)
    out0 = fwd_fxn0(img0,sims0)
    with TimeIt(timer,"fwd"):
        with MemIt(memer,"fwd"):
            out0 = fwd_fxn0(img0,sims0)
    img0.retain_grad()
    sims0.retain_grad()
    loss = out0.abs().mean()
    with TimeIt(timer,"bwd"):
        with MemIt(memer,"bwd"):
            loss.backward()

    # print(timer)
    # print(memer)

    # -- benchmark [v3] --
    img2 = img.clone().requires_grad_(True)
    fwd_fxn2 = lambda x: nat(rearrange(x,'b c h w -> b h w c'))
    out2 = fwd_fxn2(img2)
    with TimeIt(timer,"fwd_nat"):
        with MemIt(memer,"fwd_nat"):
            out2 = fwd_fxn2(img2)
    img2.retain_grad()
    loss = out2.abs().mean()
    with TimeIt(timer,"bwd_nat"):
        with MemIt(memer,"bwd_nat"):
            loss.backward()

    # -- benchmark [v2] --
    img3 = img.clone().requires_grad_(True)
    sims3 = sims.clone().requires_grad_(True)
    fwd_fxn3 = lambda x,s: sna(x,s)
    out3 = fwd_fxn1(img3,sims3)
    with TimeIt(timer,"fwd_sna"):
        with MemIt(memer,"fwd_sna"):
            out3 = fwd_fxn3(img3,sims3)
    img3.retain_grad()
    sims3.retain_grad()
    loss = out3.abs().mean()
    with TimeIt(timer,"bwd_sna"):
        with MemIt(memer,"bwd_sna"):
            loss.backward()

    print("-"*10)
    print(out0[0,0,0:5,0:5])
    print(out1[0,0,0:5,0:5])
    print(out2[0,0,0:5,0:5])
    print(out3[0,0,0:5,0:5])
    print("-"*10)
    # print(out0[0,0,3:5,3:5])
    # print(out1[0,0,3:5,3:5])
    print("-"*10)
    print(out0[0,0,8:12,8:12])
    print(out1[0,0,8:12,8:12])
    print(out2[0,0,8:12,8:12])
    print(out3[0,0,8:12,8:12])
    print("-"*10)

    print(timer)
    print(memer)

    print(out0.shape,out1.shape)
    print(th.mean((out0 - out1)**2))
    print(th.mean((img1.grad - img0.grad)**2))
    print(th.mean((sims1.grad - sims0.grad)**2))
    assert(th.mean((out0 - out1)**2)<1e-5)
    assert(th.mean((img1.grad - img0.grad)**2)<1e-5)
    assert(th.mean((sims1.grad - sims0.grad)**2)<1e-5)
    print("pass.")
    return

    # -- check attn --
    img0 = img.clone().requires_grad_(True)
    sims0 = sims.clone().requires_grad_(True)
    print("img0.shape,sims0.shape: ",img0.shape,sims0.shape)
    print(".")
    fwd_fxn = lambda x: ssna(x,sims0)
    th.autograd.gradcheck(fwd_fxn, img0, eps=1e-5, atol=1e-5,
                          nondet_tol=1e-5, raise_exception=True)
    print("..")
    fwd_fxn = lambda s: ssna(img0,s)
    th.autograd.gradcheck(fwd_fxn, sims0, eps=1e-5, atol=1e-5,
                          nondet_tol=1e-5, raise_exception=True)
    print("...")
    return

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
