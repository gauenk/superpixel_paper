

# -- basic --
import numpy as np
import torch as th
from easydict import EasyDict as edict

# -- testing --
from dev_basics.utils.misc import set_seed

# -- data --
import data_hub

# -- viz --
from einops import rearrange
from dev_basics.utils import vid_io
from skimage.segmentation import mark_boundaries
from spin.models.spin import ssn_iter as ssn_iter_spin
from spin.models.spin_stnls import ssn_iter as ssn_iter_stnls

def load_video(cfg):
    device = "cuda:0"
    data,loaders = data_hub.sets.load(cfg)
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,0,cfg.nframes)
    vid = data[cfg.dset][indices[0]]['clean'][None,:].to(device)/255.
    noisy = data[cfg.dset][indices[0]]['noisy'][None,:].to(device)/255.

    # -- down/up --
    size = list(vid.shape[-2:])
    print(vid.shape)
    vid = th.nn.functional.interpolate(vid,scale_factor=0.5,mode="bicubic")
    vid = th.nn.functional.interpolate(vid,size=size,mode="bicubic")[None,]
    print(vid.shape)

    # -- optional --
    seg = None
    if "seg" in data[cfg.dset][indices[0]]:
        seg = data[cfg.dset][indices[0]]["seg"]
    seg = seg[0]
    print(len(seg))
    print([len(s) for s in seg])
    print([len(s[0]) for s in seg])
    print(seg[0][0])
    print(seg[0][0][0][0])
    # print([s[0] for s in seg])
    # seg = th.stack([th.from_numpy(s[0].astype(np.float)) for s in seg])

    # seg = seg[0]
    # print(len(seg))
    # print(seg[0])
    print(seg.shape)
    exit()

    # -- optional --
    vid = vid[...,256:512,256:512]

    return vid,seg

def filter_labels_v0(labels):
    print(labels.shape)
    K = 5
    conv2d = th.nn.Conv2d(1,1,K,padding_mode="reflect",padding="same")
    weight = th.ones((K,K))
    weight /= weight.sum()
    conv2d.weight.data[...] = weight
    # print(conv2d.weight.data.shape)
    # print("labels.shape: ",labels.shape)
    # exit()
    labels = conv2d(th.from_numpy(labels[:,None]).float())[0].detach().numpy()
    labels = labels.round().astype(np.int)
    # print("labels.shape: ",labels.shape)
    # exit()
    return labels

def filter_labels(labels):
    ps = 7
    labels = th.from_numpy(labels)[None,:].float()
    unfold = th.nn.Unfold((ps,ps), dilation=1, padding=0, stride=1)
    print(labels.shape)
    db = unfold(labels)
    print(db.shape)
    exit()


def main():

    # -- config --
    device = "cuda:0"
    set_seed(123)
    cfg = edict()
    # cfg.dname = "set8"
    # cfg.dset = "val"
    # cfg.dname = "davis"
    cfg.dname = "bsd500"
    cfg.dset = "tr"
    # cfg.isize = "128_128"
    # cfg.dname = "iphone_sum2023"
    # cfg.dset = "tr"
    # cfg.isize = "128_128"
    # cfg.vid_name = "sunflower"
    # cfg.vid_name = "hypersmooth"
    # cfg.vid_name = "snowboard"
    # cfg.vid_name = "scooter-black"
    # cfg.vid_name = "shelf_cans"
    cfg.vid_name = "122048"
    cfg.sigma = 0.001
    cfg.nframes = 1

    # -- prepare inputs --
    ftrs,seg = load_video(cfg)
    ftrs = ftrs[:,0] # no time
    B,F,H,W = ftrs.shape
    niters = 5
    S = 8
    M = 0.0
    suffix = "s%d"%S + "_" + "m"+str(M).replace(".","p")

    # -- exec --
    sims_spin,num_spin = ssn_iter_spin(ftrs,n_iter=niters,stoken_size=[S,S],M=M)
    sims_stnls,num_stnls = ssn_iter_stnls(ftrs,n_iter=niters,stoken_size=[S,S],M=M)
    print(sims_spin.shape)
    print(sims_stnls.shape)

    # -- get labels --
    sims_spin = sims_spin.reshape(B,-1,H,W)
    sims_stnls = sims_stnls.reshape(B,-1,H,W)
    labels_spin = th.argmax(sims_spin,1).cpu().numpy()
    labels_stnls = th.argmax(sims_stnls,1).cpu().numpy()

    # -- viz --
    print(labels_spin[0,88:94,:8])
    print(labels_stnls[0,88:94,:8])
    print(labels_spin[0,-8:,:8])
    print(labels_stnls[0,-8:,:8])

    # -- filtering --
    use_filter = False
    if use_filter:
        labels_spin = filter_labels(labels_spin)
        labels_stnls = filter_labels(labels_stnls)

    # -- place segs --
    print(ftrs.shape)
    print(labels_spin.shape)
    print(labels_stnls.shape)
    ftrs = rearrange(ftrs.cpu().numpy(),'b f h w -> b h w f')
    seg_spin = mark_boundaries(ftrs[0],labels_spin[0])
    seg_spin = rearrange(seg_spin,'h w f -> 1 1 f h w')
    seg_stnls = mark_boundaries(ftrs[0],labels_stnls[0])
    seg_stnls = rearrange(seg_stnls,'h w f -> 1 1 f h w')

    # -- seg --
    # print(seg)

    # -- viz --
    vid_io.save_video(seg_spin,"output/segs/","spin_%s"%suffix)
    vid_io.save_video(seg_stnls,"output/segs/","stnls_%s"%suffix)


if __name__ == "__main__":
    main()
