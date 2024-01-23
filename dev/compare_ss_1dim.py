"""

Vizualize the difference between a 1-dim example and a 2-dim example


"""



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

# -- plotting --
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap

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

# def get_sp_grid(H,W,S):
#     h_grid = np.arange(0,H)
#     w_grid = np.arange(0,W)
#     grid = np.stack(np.meshgrid(h_grid,w_grid))
#     return grid

def superpixels_1dim(ftrs,S,lamb=10,M=0.01):
    N = len(ftrs)
    # skip = (N-1)//K+1
    # nSP = (N-1)//S+1
    ftrs = th.cat([ftrs,M*th.arange(N).to(ftrs.device)[:,None]],-1)
    sample_grid = th.arange(N)[::S]
    #nSP = len(sample_grid)
    nSP = len(sample_grid)
    # print("len(sample_grid),nSP: ",len(sample_grid),nSP,S)
    roff = [th.randint(3,(1,)).item()-1 for i in range(len(sample_grid))]
    roff[0] = 0
    roff[-1] = 0
    # print(roff)
    centroids = th.stack([ftrs[k+roff[i]] for i,k in enumerate(sample_grid)])
    # print("cshape: ",centroids.shape,N,nSP,ftrs.shape)
    niters = 10
    for iter_i in range(niters):

        # -- compute dists --
        dists = th.zeros((N,nSP),device=ftrs.device)
        ksize = int(S*2)
        for k in range(nSP):
            i = k*S
            si = max(i - ksize//2,0)
            ei = min(si + ksize,N)
            si = ei - ksize
            # print(i,si,ei,N,S)
            ftrs_k = ftrs[si:ei]
            dists_k = th.exp(-lamb*th.cdist(ftrs_k[None:],centroids[None,[k]]))[0]
            # print(dists_k.shape)
            dists[si:ei,k] = dists_k[:,0]
            # dists[:si,k] = 0
            # dists[ei:,k] = 0
        # print(dists)
        assert th.all(dists.sum(-1) > 0)

        # -- compute labels --
        labels = th.argmax(dists,1)
        # print(labels)
        # print(labels.unique())
        if iter_i >= (niters-1): break
        th.cuda.synchronize()

        # -- update --
        for k in range(nSP):
            args = th.where(labels==k)[0]
            if len(args) == 0: continue
            # print(ftrs.shape,args)
            mean_k = centroids[k].clone()#th.zeros_like(ftrs[0])
            for f in range(ftrs.shape[1]):
                mean_k[f] = th.mean(th.gather(ftrs[:,f],0,args))
            # mean_k = th.mean(th.gather(ftrs,0,args))
            # print("mean_k.shape: ",mean_k.shape,centroids[k].shape)
            centroids[k] = mean_k

    return labels

def smoothed_labels(ftrs,labels,lamb=5):
    K = len(labels.unique())
    smoothed = th.zeros_like(ftrs[:,0])
    # print("^"*20)
    ftrs = ftrs[:,0]
    # print(ftrs.shape)
    for k in range(K):
        args = th.where(labels==k)[0]
        if len(args) == 0: continue
        ftrs_k = th.gather(ftrs,0,args)[None,:,None]
        attn_k = th.cdist(ftrs_k,ftrs_k)[0]
        # print(attn_k.shape)
        smoothed_k = (-lamb*attn_k).softmax(-1) @ ftrs_k[0]
        smoothed_k = smoothed_k[:,0]
        # print(smoothed.shape,smoothed_k.shape,args.shape)
        # smoothed += th.scatter(smoothed_k,1,args)
        smoothed.scatter_add_(0,args,smoothed_k)
        # smoothed[args] = smoothed_k[:,0]
    return smoothed

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
    print("ftrs.shape: ",ftrs.shape)
    img = 0.299*ftrs[0,0] + 0.587*ftrs[0,1] + 0.114*ftrs[0,2]
    print(img.shape)
    noisy = img + (5./255.)*th.randn_like(img)

    #
    # get region --
    #
    # print(img)
    # print(img[90:100,90:100])
    print(img[93,85:115])
    # plt.imshow(img[90:100,85:115].cpu().numpy())
    plt.imshow(img.cpu().numpy())
    plt.savefig("img.png")
    plt.close("all")

    # print(img[95,85:115])
    # fig,ax = plt.subplots(1,4,figsize=(6,6))
    dpi = 200
    ginfo = {'wspace':0.1, 'hspace':0.0,
             "top":0.92,"bottom":0.16,"left":.07,"right":0.98}
             # "top":0.92,"bottom":0.14,"left":.11,"right":0.98}
    fig,ax = plt.subplots(1,4,figsize=(9,3),gridspec_kw=ginfo,dpi=200)
    # fig,ax = plt.subplots(1,4,figsize=(9,3))
    lamb = 5

    # nb = noisy[87,:100][:,None]
    # x = img[87,:100][:,None]
    nb = noisy[90,:100][:,None]
    x = img[90,:100][:,None]



    # -- clean/noisy --
    ax[0].plot(x.cpu().numpy(),'k')
    ax[0].plot(nb.cpu().numpy(),'r--')

    #
    # -- global --
    #

    # -- smoothing --
    cdist = th.cdist(nb[None,:],nb[None,:])[0]
    attn = (-lamb*cdist).softmax(-1)
    smoothed = (attn @ nb)

    # -- gt/global --
    ax[1].plot(x.cpu().numpy(),'k')
    ax[1].plot(smoothed.cpu().numpy(),'b')
    print("global: ",
          th.mean((x.cpu()-smoothed.cpu()).abs()).item(),
          th.max((x.cpu()[:,0]-smoothed.cpu()).abs()).item())



    #
    # -- "superpixel" or piecewise smoothing --
    #

    # K = 23
    S = 3
    lamb_sp = 1
    M = 0.1
    num_sp_samples = 10
    labels = superpixels_1dim(nb,S,lamb=lamb_sp,M=M)
    smoothed = smoothed_labels(nb,labels,lamb=lamb)
    for i in range(num_sp_samples-1):
        labels = superpixels_1dim(nb,S,lamb=lamb_sp,M=M)
        smoothed += smoothed_labels(nb,labels,lamb=lamb)
    smoothed = smoothed/num_sp_samples
    ax[3].plot(x.cpu().numpy(),'k')
    # ax[2].plot(smoothed.cpu().numpy(),'b')
    K = len(labels.unique())
    cols = []
    bnds = []
    # colors = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    #           'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #           'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    colors = plt.colormaps.get_cmap('autumn').resampled(K)(np.arange(K))
    ax[3].plot(smoothed.cpu().numpy(),'b-',linewidth=1)
    for k in range(K):
        args = th.where(labels==k)[0]
        sm_k = th.gather(smoothed,0,args).cpu().numpy()
        args = args.cpu().numpy()
        # ax[3].plot(args,sm_k,'-',color=colors[k%len(colors)],
        #               linewidth=1.5)
        # print(args)
        # for i in range(len(sm_k)):
        #     cols.append('r')
        #     bnds.append(sm_k-0.5)
        # # ax[2].plot(args.cpu().numpy(),sm_k,'x')
    print(x.shape,smoothed.shape)
    print("sp: ",
          th.mean((x.cpu()[:,0]-smoothed.cpu()).abs()).item(),
          th.max((x.cpu()[:,0]-smoothed.cpu()).abs()).item())
    # cols = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    #         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    # bnds = [0, 26, 31, 33, 52, 75]
    # bnds = [0, 50]
    # cmap = ListedColormap([cols[i] for i in range(len(bnds))])
    # norm = BoundaryNorm(bnds, cmap.N)
    # smoothed = smoothed.cpu().numpy()
    # segs = np.c_[np.arange(len(smoothed)),smoothed].reshape((-1,1,2))
    # segs = np.concatenate([segs[:-1], segs[1:]], axis=1)
    # print("segs.shape: ",segs.shape)
    # lc = LineCollection(segs, cmap=cmap, norm=norm)
    # # lc.set_array(dydx)
    # lc.set_linewidth(2)
    # ax[2].add_collection(lc)
    # ax[2].autoscale()


    #
    # -- neighborhood --
    #

    ksize = 6
    tofill = th.zeros(len(nb))
    for i in range(len(nb)):
        si = max(i - ksize//2,0)
        ei = min(si + ksize,len(nb))
        si = ei - ksize
        sm_i = (-lamb*cdist[si:ei,si:ei]).softmax(-1) @ nb[si:ei]
        tofill[i] = sm_i[i-si]
    ax[2].plot(x.cpu().numpy(),'k')
    ax[2].plot(tofill.cpu().numpy(),'b')

    ax[0].set_ylabel("Pixel Values",fontsize=14)
    for i in range(1,4):
        ax[i].set_yticks([])
        ax[i].set_yticklabels([])
    ax[0].set_title("Clean and Noisy",fontsize=14)
    ax[1].set_title("Global",fontsize=14)
    ax[2].set_title("Local Neigh.",fontsize=14)
    ax[3].set_title("Superpixels",fontsize=14)

    # for i in range(4):
    #     ax[i].set_xlabel("Pixel Index",fontsize=14)
    fig.supxlabel("Pixel Index",fontsize=14)

    print(x.shape,smoothed.shape)
    print("nb: ",
          th.mean((x.cpu()[:,0]-tofill.cpu()).abs()).item(),
          th.max((x.cpu()[:,0]-tofill.cpu()).abs()).item())

    plt.savefig("tmp.png")


if __name__ == "__main__":

    print("hi.")
    main()


