"""

A script to highlight regions from superpixel examples

"""


# -- data mng --
import pandas as pd
import pprint
pp = pprint.PrettyPrinter(depth=5,indent=8)
import copy
dcopy = copy.deepcopy

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat
import cache_io

# -- vision --
import cv2
from PIL import Image
from torchvision.utils import make_grid,save_image
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
from torchvision.transforms.functional import center_crop

# -- dev basics --
from dev_basics.utils.misc import optional
from dev_basics.utils.metrics import compute_psnrs,compute_ssims

# -- data io --
import data_hub

# -- management --
from pathlib import Path
from easydict import EasyDict as edict

# -- plotting --
from matplotlib import pyplot as plt

SAVE_DIR = Path("./output/sp_examples/")


def read_name(name,smooth=True):
    keys = ["ref","sna_s10","seg_s0","ssn_s0","mix_s10","ssn_s10"]
    info = {"ref":"data/sr/BSD500/images/test/%s",
            "sna_s10":"output/eval_superpixels/s_14/ssna-m-deno1-ssn0-seg-s10_again/%s",
            "seg_s0":"output/eval_superpixels/s_14/ssna-m-deno0-ssn1-seg-s0_again/%s",
            "ssn_s0":"output/eval_superpixels/s_14/ssna-m-deno0-ssn1-pix-s0_again/%s",
            "mix_s10":"output/eval_superpixels/s_14/ssna-m-deno1-ssn1-pix-s10_again/%s",
            "ssn_s10":"output/eval_superpixels/s_14/ssna-m-deno0-ssn1-pix-s10_again/%s",
            # "ssn_s0":"output/eval_superpixels/s_14/ssn/%s",
            # "mix_s10":"output/eval_superpixels/s_14/ssna-m-deno1-ssn1-s10/%s",
            # "ssn_s10":"output/eval_superpixels/s_14/ssna-m-deno0-ssn1-s10/%s"
    }
    imgs = []
    for key in keys:
        fn = info[key]%name
        is_ref = key == "ref"
        if is_ref: fn = fn+".jpg"
        else: fn = fn+"_s.jpg" if smooth else fn+".jpg"
        img = Image.open(fn).convert("RGB")
        img = np.array(img).transpose(2,0,1)/255.
        img = th.from_numpy(img)

        size = 256

        if smooth:
            img = center_crop(img,(size,size))
        else:
            sH,sW = 300,250
            if is_ref:
                img = img[:,sH//2:sH//2+size//2,sW//2:sW//2+size//2]
            else:
                img = img[:,sH:sH+size,sW:sW+size]
        # if smooth:
        # else:
        #     img = center_crop(img[:,70:],(size,size))


        if smooth:
            img = TF.resize(img,(156,156),InterpolationMode.NEAREST)
        else:
            img = TF.resize(img,(156,156),InterpolationMode.BILINEAR)
        imgs.append(img)
    _,H,W = imgs[-1].shape
    imgs = th.stack([i[:,:H,:W] for i in imgs])
    print("imgs.shape: ",imgs.shape)
    grid = make_grid(imgs,nrow=len(imgs),padding=1)

    info = []
    for ix,key in enumerate(keys):
        info_i = {}
        if key == "ref":
            info_i['psnr'] = 0
            info_i['ssim'] = 0
        else:
            info_i['psnr'] = compute_psnrs(imgs[0][None,None],
                                           imgs[ix][None,None],div=1.)[0].item()
            info_i['ssim'] = compute_ssims(imgs[0][None,None],
                                           imgs[ix][None,None],div=1.)[0].item()
        info_i['key'] = key
        info_i['name'] = name
        info.append(info_i)
    return grid,info

def main():

    # -- create denos --
    # names = ["35028","103029","246009"]
    names = ["35028","103029"]
    info = []

    # -- smoothed grids --
    grids = []
    for name in names:
        grid_i,info_i = read_name(name,smooth=True)
        grids.append(grid_i)
        info.extend(info_i)
    grid_sm = make_grid(grids,nrow=1,padding=0)
    info = pd.DataFrame(info)
    print(info)
    # exit()

    # -- spix grids --
    names = ["35028"]
    grids = []
    for name in names:
        grid_i,_ = read_name(name,smooth=False)
        grids.append(grid_i)
    grid_sp = make_grid(grids,nrow=1,padding=0)

    # -- format grids --
    print(grid_sm.shape,grid_sp.shape)
    grids = [grid_sm,grid_sp]
    # grid = make_grid(grids,nrow=1,padding=0)
    grid = th.cat(grids,1)
    grid = grid_sm

    # -- save --
    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir(parents=True)
    fn = SAVE_DIR/("%s_%d.png" % ("grid",0))
    print(fn)
    print(grid.shape)
    save_image(grid,fn)

if __name__ == "__main__":
    main()
