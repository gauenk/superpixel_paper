

import numpy as np
import torch as th
from PIL import Image
from pathlib import Path
from dev_basics.utils import vid_io
from torchvision import utils as tv_utils
from einops import rearrange

# def get_file_dir(root,subdir):
#     adir = None
#     for sub in subdir:
#         dir_i = Path(root)/sub
#         if dir_i.exists():
#             adir = dir_i
#     assert not(adir is None)
#     return adir

# def compare_example(fn_a,fn_b,fn_gt,sname):

#     # -- open image --
#     name = fn_a.name
#     a = np.array(Image.open(fn_a).convert("RGB"))/255.
#     b = np.array(Image.open(fn_b).convert("RGB"))/255.
#     gt = np.array(Image.open(fn_gt).convert("RGB"))/255.
#     # print(a.shape,b.shape)

#     # -- delta --
#     H,W,C = a.shape
#     delta_a = np.abs((a - gt[:H,:W]).mean(-1))
#     delta_b = np.abs((b - gt[:H,:W]).mean(-1))
#     # delta_a = a
#     # delta_b = b

#     # -- params --
#     size = 64
#     # thresh = 1.25e-2
#     thresh = 1.5
#     thresh_1 = 0.20

#     # -- compare --
#     nH,nW = (H)//size,(W)//size
#     for hi in range(nH):
#         for wi in range(nW):
#             patch_a = delta_a[size*hi:size*(hi+1),size*wi:size*(wi+1)]
#             patch_b = delta_b[size*hi:size*(hi+1),size*wi:size*(wi+1)]
#             # delta_ab = np.abs(patch_a - patch_b).mean()
#             delta_ab = np.mean(1.*(patch_a < patch_b))
#             delta_ab_r = np.mean(1.*(patch_a > patch_b))
#             # if delta_ab > thresh:
#             # print(delta_ab,delta_ab_r)
#             if (delta_ab / delta_ab_r > thresh) and (delta_ab > thresh_1):
#                 print(name,fn_gt,size*hi,size*wi,delta_ab,delta_ab_r)

#                 # -- save --
#                 a_ij = a[size*hi:size*(hi+1),size*wi:size*(wi+1)]
#                 b_ij = b[size*hi:size*(hi+1),size*wi:size*(wi+1)]
#                 gt_ij = gt[size*hi:size*(hi+1),size*wi:size*(wi+1)]
#                 save_example(sname,fn_gt.stem,a_ij,b_ij,gt_ij,hi,wi)

# def save_example(sname,name,a_ij,b_ij,gt_ij,hi,wi):
#     outdir = Path("output/find_examples/%s/%s/" % (sname,name))
#     if not outdir.exists():
#         outdir.mkdir(parents=True)
#     # print(outdir)
#     loc = "%d_%d" % (hi,wi)
#     a_ij = np.uint8(np.clip(a_ij*255,0,255.))
#     b_ij = np.uint8(np.clip(b_ij*255,0,255.))
#     gt_ij = np.uint8(np.clip(gt_ij*255,0,255.))

#     Image.fromarray(a_ij).save(outdir / ("%s_stnls.png"%loc))
#     Image.fromarray(b_ij).save(outdir / ("%s_default.png"%loc))
#     Image.fromarray(gt_ij).save(outdir / ("%s_gt.png"%loc))

# def compare_files(a,b,gt,sname):
#     files = sorted(list(gt.iterdir()))
#     for i,fn in enumerate(files):
#         fn_a = a/("%d_x2.png"%(i+1))
#         fn_b = b/("%d_x2.png"%(i+1))
#         assert fn_a.exists(),fn_a
#         assert fn_b.exists(),fn_b
#         compare_example(fn_a,fn_b,fn,sname)

def get_boxes(fn_a,fn_b,box,W):
    img_a = np.array(Image.open(fn_a))
    img_b = np.array(Image.open(fn_b))
    crop_a = img_a[box[0]:box[0]+W,box[1]:box[1]+W]
    crop_b = img_b[box[0]:box[0]+W,box[1]:box[1]+W]
    crop_a = rearrange(th.from_numpy(crop_a),'h w c -> c h w')
    crop_b = rearrange(th.from_numpy(crop_b),'h w c -> c h w')
    return crop_a,crop_b

def main():

    midfix_a = "ca421/epoch=49/"
    midfix_b = "54b6a/epoch=49/"
    base = "output/deno/test/"
    gt = "data/benchmarks/"
    W = 128
    sigmas = [25,30,35]
    boxes = {
        "set5":{"5_x1":[50,50]},
        "set14":{"2_x1":[100,420],
                      # "5_x1":[100,100]
                      #"11_x1":[100,250]
             },
        "u100":{#"96_x1":[400,800],
                "5_x1":[600,500],
        }
    }
    grids = []
    for dname in boxes.keys():
        for iname,box in boxes[dname].items():
            imgs = []
            for sigma in sigmas:
                fn_a = Path(base)/str(sigma)/midfix_a/dname/("%s.png"%iname)
                fn_b = Path(base)/str(sigma)/midfix_b/dname/("%s.png"%iname)
                img_a,img_b = get_boxes(fn_a,fn_b,box,W)
                imgs.append(img_b)
                imgs.append(img_a)
            imgs = rearrange(th.stack(imgs),'(a b) c h w -> (b a) c h w',b=2)
            grid = tv_utils.make_grid(imgs,nrow=len(sigmas))
            grids.append(grid)

    # -- get grid --
    # grid_a = tv_utils.make_grid(imgs_a,nrow=len(sigmas))
    # grid_b = tv_utils.make_grid(imgs_b,nrow=len(sigmas))
    grid = tv_utils.make_grid(grids,nrow=3)
    vid_io.save_image(grid,"output/figures/examples/grid")


if __name__ == "__main__":
    main()
