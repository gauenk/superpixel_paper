
import torch as th
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries

from einops import rearrange,repeat
from torchvision.utils import draw_bounding_boxes,draw_segmentation_masks

# -- nicer viz --
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

def get_sample():
    # iname = "35028"
    # iname = "302003"
    iname = "126039"
    # fn = Path("../BASS/pytorch_version_og/out_s123/csv/%s.csv" % iname)
    fn = Path("../BASS/pytorch_version_og/out_img/csv/%s.csv" % iname)
    sp = np.genfromtxt(fn, delimiter=',').astype(np.int)
    fn = Path("../BASS/images/%s.jpg" % iname)
    img = np.array(Image.open(fn))
    return img,sp

def viz_sp(img,sp,sp_id):

    # -- args --
    args = np.where(sp == sp_id)
    hgrid,wgrid = args[0],args[1]

    # -- get surrounding region --
    min_h,max_h = np.min(hgrid)-5,np.max(hgrid)+5
    min_w,max_w = np.min(wgrid)-5,np.max(wgrid)+5
    hmid = (max_h + min_h)//2
    wmid = (max_w + min_w)//2

    # -- crop --
    img_c = img[min_h:max_h,min_w:max_w]
    sp_c = sp[min_h:max_h,min_w:max_w]
    args = np.where(sp_c == sp_id)
    args = np.where(sp_c != sp_id)

    # -- alpha only around sp --
    # alphas = np.zeros(img_c[:,:,:1].shape)
    # alphas[args] = 255
    # img_c = np.concatenate([img_c,alphas.astype(np.uint8)],-1)

    # -- zero non-sp pixels --
    # alphas = np.zeros(img_c[:,:,:1].shape)
    # alphas[args] = 255
    img_c[args] = 0


    return img_c,hmid,wmid,sp_id

def viz_sp_square(img,sp,sp_id,hmid,wmid,ksize,vsize):

    # -- img --
    sh,sw = hmid-vsize//2,wmid-vsize//2
    eh,ew = sh+vsize,sw+vsize
    img_c = img[sh:eh,sw:ew]
    sp_c = sp[sh:eh,sw:ew]
    og = img_c

    # -- recompute mid --
    argsH,argsW = np.where(sp_c == sp_id)
    hmid = np.mean(argsH)
    wmid = np.mean(argsW)

    # -- zero non-superpixel regions --
    args = np.where(sp_c != sp_id)
    alphas = np.zeros(img_c[:,:,:1].shape)
    # alphas[args] = 255
    # img_c = np.concatenate([img_c,alphas.astype(np.uint8)],-1)
    img_c[args] = 0

    # -- viz box --
    sh,sw = hmid-ksize//2,wmid-ksize//2
    eh,ew = sh+ksize,sw+ksize
    kbox = [sh,sw,eh,ew]
    # img_c = viz_loc_sp_seg(img_c,sp_c==sp_id)
    img_c = viz_loc_box(img_c,kbox)


    return img_c

def viz_loc_box(img,box):
    img = rearrange(img,'h w c -> c h w')
    img = th.from_numpy(img).type(th.uint8)
    box = th.tensor(box)
    img = draw_bounding_boxes(img,box[None,:],fill=True,colors="red")
    img = rearrange(img.numpy(),'c h w -> h w c')
    return img

def viz_loc_sp_seg(img,seg):
    img = rearrange(img,'h w c -> c h w')
    img = th.from_numpy(img).type(th.uint8)
    seg = th.tensor(seg)
    print(img.shape,seg.shape)
    img = draw_segmentation_masks(img, seg, alpha=0.8, colors="blue")
    # img = draw_bounding_boxes(img,box[None,:],fill=True,colors="#FF00FF")
    img = rearrange(img.numpy(),'c h w -> h w c')
    return img

def viz_square(img,hmid,wmid,ksize,vsize):

    # -- get box --
    sh,sw = hmid-vsize//2,wmid-vsize//2
    eh,ew = sh+vsize,sw+vsize
    img_c = img[sh:eh,sw:ew]
    hmid = img_c.shape[0]//2
    wmid = img_c.shape[1]//2

    # -- viz box --
    sh,sw = hmid-ksize//2,wmid-ksize//2
    eh,ew = sh+ksize,sw+ksize
    kbox = [sh,sw,eh,ew]
    img_c = viz_loc_box(img_c,kbox)

    return img_c

def irz(img,tH,tW):
    ndim = img.ndim
    s0 = img.shape[-1]
    if img.ndim == 2:
        img = img[None,:]
    elif img.ndim == 3 and s0 == 3:
        img = rearrange(img,'h w c -> c h w')

    is_t = th.is_tensor(img)
    if not is_t: img = th.from_numpy(img)
    img = TF.resize(img,(tH,tW),
                    interpolation=InterpolationMode.NEAREST)
    if not is_t: img = img.numpy()

    if ndim == 2:
        img = img[0]
    elif img.ndim == 3 and s0 == 3:
        img = rearrange(img,'c h w -> h w c')

    return img

def topk_sampling(sp_img,K):
    args = np.where(sp_img[:,:,-1]>0)
    coords = np.stack(args).T
    C = len(coords)
    sel = np.random.choice(np.arange(C),size=C-K,replace=False)
    # print(coords)
    # print(sel)
    for sel_ix in sel:
        sp_img[coords[sel_ix,0],coords[sel_ix,1]] = 0
    return sp_img

def main():

    img,sp = get_sample()
    print(img.shape)
    print(sp.shape)

    W = 96
    hi,wi = 115,130
    sh,sw = hi-W//2,wi-W//2
    eh,ew = sh+W,sw+W
    img = irz(img[sh:eh,sw:ew],96*2,96*2)
    sp = irz(sp[sh:eh,sw:ew],96*2,96*2)
    sp_l = irz(sp,96*4,96*4)
    sp_id = sp[64,64]

    # -- viz image and seg --
    img_v = viz_loc_sp_seg(img.copy(),sp==sp_id)
    Image.fromarray(img_v).save("viz_region.png")
    seg = mark_boundaries(img,sp,mode='subpixel',color=[1,0,0])
    seg = np.clip(seg * 255,0,255.).astype(np.uint8)
    print("seg.shape: ",seg.shape,sp_l.shape)
    seg = viz_loc_sp_seg(seg.copy(),sp_l[:-1,:-1]==sp_id)
    Image.fromarray(seg).save("viz_seg.png")


    # -- viz a single sp --
    sp_img,hmid,wmid,sp_id = viz_sp(img.copy(),sp,sp_id)
    print(sp_img.shape)
    Image.fromarray(irz(sp_img,128,128)).save("viz_sp.png")

    # -- viz top-k sp --
    K = 300
    sp_img,hmid,wmid,sp_id = viz_sp(img.copy(),sp,sp_id)
    sp_img = topk_sampling(sp_img.copy(),K)
    Image.fromarray(irz(sp_img,128,128)).save("viz_ksp.png")

    # - -viz local region --
    ksize = 22
    vsize = 55
    loc_img = viz_sp_square(img.copy(),sp,sp_id,hmid,wmid,ksize,vsize)
    Image.fromarray(irz(loc_img,128,128)).save("viz_sp_loc.png")

    ksize = 22
    vsize = 55
    loc_img = viz_square(img.copy(),hmid,wmid,ksize,vsize)
    Image.fromarray(irz(loc_img,128,128)).save("viz_loc.png")



if __name__ == "__main__":
    main()

