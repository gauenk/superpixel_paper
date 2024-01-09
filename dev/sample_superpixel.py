

# -- basic --
import numpy as np
import torch as th
from easydict import EasyDict as edict

# -- testing --
from dev_basics.utils.misc import set_seed

# -- nicer viz --
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision import utils as tv_utils

# -- data --
import data_hub

# -- nicer viz --
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# -- viz --
from einops import rearrange
from dev_basics.utils import vid_io
from skimage.segmentation import mark_boundaries
from spin.models.spin import ssn_iter as ssn_iter_spin
from spin.models.spin import calc_init_centroid
from spin.models.spin_stnls import ssn_iter as ssn_iter_stnls

def load_video(cfg):
    device = "cuda:0"
    data,loaders = data_hub.sets.load(cfg)
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,0,cfg.nframes)
    vid = data[cfg.dset][indices[0]]['clean'][None,:].to(device)/255.
    noisy = data[cfg.dset][indices[0]]['noisy'][None,:].to(device)/255.

    # -- down/up --
    size = list(vid.shape[-2:])
    vid = th.nn.functional.interpolate(vid[0],scale_factor=0.5,mode="bicubic")
    vid = th.nn.functional.interpolate(vid,size=size,mode="bicubic")[None,]

    # -- optional --
    seg = None
    if "seg" in data[cfg.dset]:
        seg = data[cfg.dset][indices[0]]["seg"]

    # -- optional --
    # vid = vid[...,256:512,256:512]

    return vid,seg

def load_chick(cfg):
    device = "cuda:0"
    vid = vid_io.read_video("data/crop_cat_chicken")[None,:]/255.
    vid = vid.to(device)
    seg = None
    vid = vid[...,62:62+156,120:120+156]
    print("vid.shape: ",vid.shape)
    return vid,seg

def irz(img,tH,tW):
    is_t = th.is_tensor(img)
    if not is_t: img = th.from_numpy(img)
    img = TF.resize(img,(tH,tW),
                    interpolation=InterpolationMode.NEAREST)
    if not is_t: img = img.numpy()
    return img

def viz_centroid(ftrs,S,ws,i,j,name):
    F = ftrs.shape[-1]
    cH,cW = i*S,j*S
    sH,eH = cH-(ws-1)//2,cH-(ws-1)//2+ws
    sW,eW = cW-(ws-1)//2,cW-(ws-1)//2+ws
    ftrs_og = rearrange(th.from_numpy(ftrs[0,sH:eH,sW:eW]),'h w c -> c h w').clone()
    centroid0 = rearrange(th.from_numpy(ftrs[0,sH:eH,sW:eW]),'h w c -> c h w').clone()
    attn_sims = centroid0.reshape(F,ws*ws).T @ centroid0.reshape(F,ws*ws)
    attn_sims = th.softmax(attn_sims,-1).reshape(ws,ws,ws,ws)
    attn_sims_w = attn_sims/attn_sims.max()

    info = [[3,6],[-12,-10]]
    for ix,(i,j) in enumerate(info):
        sims_cc = attn_sims_w[i,j]
        # print(sims_cc.max(),sims_cc.min())
        # # sims_cc /= sims_cc.max()
        vid_io.save_image(irz(sims_cc[None,:],64,64),
                          "output/explain_rewieghting/sims_%s_%d" % (name,ix))
        # wc_cc = (attn_sims[None,i,j]/attn_sims[None,i,j].sum()) * ftrs_og
        # wc_cc = (attn_sims[None,i,j]/attn_sims[None,i,j].sum())
        wc_cc = attn_sims[None,i,j]/attn_sims[None,i,j].max()
        wc_cc = th.exp(-10.*(1-wc_cc))
        # wc_cc /= wc_cc.max()
        wc_cc = th.cat([ftrs_og,wc_cc],0)
        vid_io.save_image(irz(wc_cc,64,64),
                          "output/explain_rewieghting/wc_%s_%d" % (name,ix))
        cx = ix if ix == 0 else 2
        centroid0[:,i,j] = 0
        centroid0[cx,i,j] = 1.
    vid_io.save_image(irz(centroid0,64,64),
                      "output/explain_rewieghting/centroid_%s" % (name))


def viz_weights(sims,ftrs):
    # -- get labels --
    a,b = 50,50
    H,W,F = ftrs.shape
    sims = sims.reshape(-1,H*W)
    N = 1000
    weights = th.zeros_like(ftrs[:,:,0]).reshape(H*W)
    print("sims.shape: ",sims.shape)
    for i in range(N):

        # -- get superpixel --
        labels = th.multinomial(sims.T,num_samples=1)
        lid = labels.reshape(H,W)[a,b]
        args = th.where(labels == lid)[0]

        # -- compute attn --
        sup_pix = th.gather(ftrs.reshape(H*W,F),0,args[:,None].expand(-1,F))
        attn = th.softmax(sup_pix @ ftrs[a,b],0)

        # -- scatter to matrix --
        # ones = th.ones_like(attn)
        weights = weights.scatter_add(0,args,attn.abs())

    # -- normalize --
    weights = weights.reshape(H,W)
    weights = weights/N
    weights /= weights.max()
    # vid_io.save_image(irz(centroid0,64,64),
    #                   "output/explain_rewieghting/centroid_%s" % (name))
    vid_io.save_image(weights[None,:,:],"output/viz_attn_weights/weights")


def main():

    # -- config --
    device = "cuda:0"
    set_seed(123)
    cfg = edict()
    cfg.dname = "set8"
    cfg.dset = "te"
    # cfg.dname = "davis"
    # cfg.dset = "val"
    # cfg.dname = "iphone_sum2023"
    # cfg.dname = "bsd500"
    # cfg.dset = "tr"
    cfg.isize = "128_128"
    cfg.vid_name = "sunflower"
    # cfg.vid_name = "hypersmooth"
    # cfg.vid_name = "snowboard"
    # cfg.vid_name = "scooter-black"
    # cfg.vid_name = "shelf_cans"
    # cfg.vid_name = "122048"
    cfg.sigma = 0.001
    cfg.nframes = 1

    # -- prepare inputs --
    # ftrs,seg = load_video(cfg)
    ftrs,seg = load_chick(cfg)
    ftrs = ftrs[:,0] # no time
    B,F,H,W = ftrs.shape
    HW = H*W
    niters = 10
    # S = 12
    S = 10
    nH,nW = (H-1)//S+1,(W-1)//S+1
    M = 0.2
    # ws = 28
    ws = 24
    # affinity_softmax = 20.
    affinity_softmax = 30.
    suffix = "s%d"%S + "_" + "m"+str(M).replace(".","p")
    # print("ftrs.shape: ",ftrs.shape)

    # -- exec --
    sims,num = ssn_iter_stnls(ftrs,n_iter=niters,stoken_size=[S,S],M=M,ws=ws,
                              affinity_softmax=affinity_softmax)

    # -- setup --
    ftrs = rearrange(ftrs,'1 f h w -> h w f')
    sims = sims.reshape(-1,H,W)
    labels = th.argmax(sims,0).cpu().numpy()

    # -- pick pixel --
    i,j = H//2+7,W//2-25 # center pixel
    VW = 64 # viz window size
    sH,eH = i-VW//2,i+VW//2
    sW,eW = j-VW//2,j+VW//2
    a,b = VW//2,VW//2
    a_list,b_list = [32,28,60],[10,50,30]
    N = 5 # number of samples
    mode = "inner" # how superpixels are marked
    alpha_base = 0.15 # transparency of non-superpixel region
    sims_w = sims[:,sH:eH,sW:eW]
    ftrs_w = ftrs[sH:eH,sW:eW]
    iftrs = th.cat([ftrs_w,th.ones_like(ftrs_w[:,:,[0]])],-1)
    iftrs = rearrange(iftrs.cpu(),'h w f -> f h w')
    superpixels = [iftrs]
    for n in range(N):

        # -- sample --
        if n == 0:
            sample = th.argmax(sims_w,0)
            # sample = th.multinomial(sims_w.reshape(-1,VW*VW).T,num_samples=1)
        else:
            sample = th.multinomial(sims_w.reshape(-1,VW*VW).T,num_samples=1)
        # print(sample,sample[a,b])
        sample = sample.reshape(VW,VW)

        bools = None
        for a,b in zip(a_list,b_list):
            # for i in range(2):
            #     for j in range(2):
            #         if bools is None:
            #             bools = sample == sample[a-1+i,b-1+j]
            #         else:
            #             bools = th.logical_or(bools,sample == sample[a-1+i,b-1+j])
            if bools is None:
                bools = sample == sample[a,b]
            else:
                bools = th.logical_or(bools,sample == sample[a,b])
        args = th.where(bools)

        # -- mark --
        # seg = mark_boundaries(ftrs_w.cpu().numpy(),sample.cpu().numpy(),mode=mode)

        # -- transparency --
        # seg = th.from_numpy(f)
        seg = ftrs_w.cpu()
        alpha = th.ones_like(seg[:,:,[0]])*alpha_base
        alpha[args] = 1.
        seg = th.cat([seg,alpha],-1)
        seg = rearrange(seg.numpy(),'h w f -> f h w')
        superpixels.append(th.from_numpy(seg))
    grid = tv_utils.make_grid(superpixels)
    vid_io.save_image(grid,"output/figures/sample_superpixel/examples")


if __name__ == "__main__":
    main()
