

# -- basic --
import numpy as np
import torch as th
from easydict import EasyDict as edict

# -- testing --
from dev_basics.utils.misc import set_seed

# -- nicer viz --
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

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
    vid = vid[...,55:55+156,120:120+156]
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


def get_sp_grid(H,W,S):
    h_grid = np.arange(0,H,S)
    w_grid = np.arange(0,W,S)
    grid = np.stack(np.meshgrid(h_grid,w_grid))
    return grid

def get_local_inds(labels,a,b,H,W,S,SW):
    # -- get superpixel center --
    labels = labels.reshape(H,W)
    lid = labels[a,b].item()
    h_center = S*(lid // (W//S))
    w_center = S*(lid % (W//S))

    # -- get grids --
    h_grid = th.arange(h_center - SW//2,h_center - SW//2+SW)
    w_grid = th.arange(w_center - SW//2,w_center - SW//2+SW)
    grid = th.stack(th.meshgrid(h_grid,w_grid))
    args = grid[0]*W + grid[1]
    return args.ravel().to(labels.device)

def get_sp_args(args,labels,a,b,H,W,S,SW):

    # -- get superpixel center --
    nH,nW = (H-1)//S+1,(W-1)//S+1
    labels = labels.reshape(H,W)
    lid = labels[a,b].item()
    h_center = S*(lid // nW)
    w_center = S*(lid % nW)
    # print(lid // nW,lid % nW,lid)

    # -- translate to smaller square --
    argsH = args // W
    argsW = args % W
    # print(h_center,w_center)
    # print(th.stack([argsH,argsW]).T)

    argsH = argsH - h_center + SW//2
    argsW = argsW - w_center + SW//2
    # print("argsH.min(),argsW.min(): ",argsH.min(),argsW.min())
    # print(th.stack([argsH,argsW]).T)
    args = argsH*SW+argsW
    return args.long()

def viz_weights(sims,ftrs,S):
    # -- get labels --
    H,W,F = ftrs.shape
    # a,b = H//2+7,W//2-25 # center pixel
    a,b = H//2-10,W//2 # center pixel
    sims = sims.reshape(-1,H*W)
    # print(a,b)
    N = 1
    SW = 2*S
    # weights = th.zeros_like(ftrs[:,:,0]).reshape(H*W)
    spix_f = th.zeros_like(ftrs).reshape(H*W,F)
    space = th.zeros((SW*SW,SW*SW),device=sims.device)
    spix = th.zeros((SW,SW,F),device=sims.device)
    lamb = 20.#1./F*1./S
    # print(ftrs.shape)
    # print("sims.shape: ",sims.shape)
    for i in range(N):

        # -- get superpixel --
        # labels = th.multinomial(sims.T,num_samples=1)
        labels = th.argmax(sims,0)
        print(labels.shape)
        lid = labels.reshape(H,W)[a,b]
        args = th.where(labels == lid)[0]
        # print("args.shape: ",args.shape)
        # print(args)
        # args = get_local_inds(labels,a,b,H,W,S,SW)
        th.cuda.synchronize()
        sp_args = get_sp_args(args,labels,a,b,H,W,S,SW)
        th.cuda.synchronize()
        # print(sp_args)

        #
        # -- Vizualize Superpixel Attention --
        #

        sup_pix = th.gather(ftrs.reshape(H*W,F),0,args[:,None].expand(-1,F))
        attn = th.softmax(lamb * (sup_pix @ sup_pix.T),1)
        print(attn)
        th.cuda.synchronize()
        print(attn.min(),attn.max())
        print(th.mean(attn),th.var(attn))
        # attn = attn/attn.max()
        # vid_io.save_image(attn[None,:],"output/viz_attn_weights/attn")

        # -- scatter to search space --
        attn_args = th.stack(th.meshgrid(sp_args,sp_args))
        attn_args = attn_args[0]*SW*SW + attn_args[1]
        # print(attn_args.min(),attn_args.max())
        # print(space.shape,attn_args.shape,attn.shape)
        # exit()
        space = space.ravel().scatter_add(0,attn_args.ravel(),attn.ravel()).reshape(SW*SW,SW*SW)
        th.cuda.synchronize()
        exact_max = space.max().item()
        print("Exact Maximum/Minimum: ",exact_max)
        space = space/exact_max # viz rewight
        vid_io.save_image(space[None,:],"output/viz_attn_weights/attn")
        print("attn_args.shape: ",attn_args.shape)
        th.cuda.synchronize()

        #
        # -- Vizualize Re-Weighted Superpixel Attention --
        #

        # -- compute normalization using the maximum LOCAL patch --
        local_args = get_local_inds(labels,a,b,H,W,S,SW)
        local_pix = th.gather(ftrs.reshape(H*W,F),0,local_args[:,None].expand(-1,F))
        attn_normz = th.sum(th.exp(lamb*(sup_pix @ sup_pix.T)),1,keepdim=True)
        # print(attn_normz)
        attn_normz = th.sum(th.exp(lamb*(local_pix @ local_pix.T)),1,keepdim=True)
        # print(attn_normz)

        # -- scatter unnormalized attn weights to search space --
        attn_rw = th.exp(lamb*(sup_pix @ sup_pix.T))#/attn_normz
        attn_args = th.stack(th.meshgrid(sp_args,sp_args))
        attn_args = attn_args[0]*SW*SW + attn_args[1]
        space[...] = 0.
        space = space.ravel().scatter_add(0,attn_args.ravel(),attn_rw.ravel()).reshape(SW*SW,SW*SW)
        space = space / attn_normz # re-weight!
        ones = th.ones_like(space[0])
        # space = space / th.maximum(th.sum((space>0),1,keepdim=True),ones)
        print("Re-Weighted Maximum: ",space.max().item())
        space = space/exact_max # viz rewight
        print(space.max())
        # space = space/space.max()
        vid_io.save_image(space[None,:],"output/viz_attn_weights/attn_rw")
        th.cuda.synchronize()


        # -- scatter feature to view superpixel --
        # sp_args = sp_args.view(-1,F).expand(-1,F)
        # print(sp_args.shape,sup_pix.shape)
        # spix = spix.view(-1,F).scatter_add(0,sp_args,sup_pix).reshape(SW,SW,F)
        # spix = rearrange(spix,'h w f -> f h w')
        # vid_io.save_image(spix,"output/viz_attn_weights/spix")

        # -- scatter feature to view superpixel --
        sp_args = sp_args.view(-1,1).expand(-1,F)
        # print(sp_args.shape,sup_pix.shape)
        spix = spix.view(-1,F).scatter_add(0,sp_args,sup_pix).reshape(SW,SW,F)
        # spix_f = spix_f.view(-1,F).scatter_add(0,sp_args,sup_pix).reshape(H,W,F)
        spix = rearrange(spix,'h w f -> f h w')
        vid_io.save_image(spix,"output/viz_attn_weights/spix")

    # -- normalize --
    # weights = weights.reshape(H,W)
    # weights = weights/N
    # weights /= weights.max()
    # vid_io.save_image(irz(centroid0,64,64),
    #                   "output/explain_rewieghting/centroid_%s" % (name))
    # vid_io.save_image(weights[None,:,:],"output/viz_attn_weights/weights")


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
    S = 8
    nH,nW = (H-1)//S+1,(W-1)//S+1
    M = 0.2
    ws = 15
    affinity_softmax = 1.
    suffix = "s%d"%S + "_" + "m"+str(M).replace(".","p")
    # print("ftrs.shape: ",ftrs.shape)

    # -- exec --
    sims,num = ssn_iter_stnls(ftrs,n_iter=niters,stoken_size=[S,S],M=M,ws=ws,
                              affinity_softmax=affinity_softmax)

    # -- get labels --
    sims = sims.reshape(B,-1,H,W)
    labels = th.argmax(sims,1).cpu().numpy()

    # -- place segs --
    # print(ftrs.shape)
    mode = "inner"
    ftrs = rearrange(ftrs.cpu().numpy(),'b f h w -> b h w f')
    seg = mark_boundaries(ftrs[0],labels[0],mode=mode)
    seg = rearrange(seg,'h w f -> 1 1 f h w')

    # -- center a centroid --
    # cH,cW = 8*S,3*S
    i,j = 8,3
    viz_centroid(ftrs.copy(),S,ws,i,j,"a")
    i,j = 7,2
    viz_centroid(ftrs.copy(),S,ws,i,j,"b")
    # viz_centroid(i,j)

    # -- viz --
    # # print("suffix: ",suffix)
    vid_io.save_video(seg,"output/segs/","stnls_%s"%suffix)
    # save_sp(ftrs,labels)
    cat = rearrange(ftrs[0],'h w c -> c h w')
    # print("cat.shape: ",cat.shape)
    # vid_io.save_image(cat,"output/explain_rewieghting/cat")


    # --  --
    viz_weights(sims[0],th.from_numpy(ftrs[0]).to(device),S)

if __name__ == "__main__":
    main()
