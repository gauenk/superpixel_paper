

import numpy as np
import torch as th
from PIL import Image
from pathlib import Path
from dev_basics.utils import vid_io
from torchvision import utils as tv_utils
from einops import rearrange
import pandas as pd

# -- nicer viz --
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


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

def get_gt(dname,iname):
    # -- read filename --
    if dname == "b100":
        dname = "bsd100"
    base = Path("/srv/disk3tb/home/gauenk/data/") / dname / "HR/"
    files = sorted(list(base.iterdir()))
    index = int(iname.split("_")[0])-1
    fn = files[index]
    print(fn.name)
    return fn

def get_boxes(fn_a,box,W):
    img_a = np.array(Image.open(fn_a))
    crop_a = img_a[box[0]:box[0]+W,box[1]:box[1]+W]
    crop_a = rearrange(th.from_numpy(crop_a),'h w c -> c h w')
    return crop_a

def irz(img,tH,tW):
    is_t = th.is_tensor(img)
    if not is_t: img = th.from_numpy(img)
    img = TF.resize(img,(tH,tW),
                    interpolation=InterpolationMode.NEAREST)
    if not is_t: img = img.numpy()
    return img

def comp_psnrs(gt,img_m):
    return -10 * th.log10(th.mean((gt/255. - img_m/255.)**2)).item()

def main():

    # "d84c37,48af63,c19c38"
    midfix_a = "48af6/epoch=49/" # biased
    midfix_b = "d84c3/epoch=49/" # aspa
    midfix_c = "c19c3/epoch=49/" # mle
    # midfixes = [midfix_a,midfix_c,midfix_b]
    method_names = ["biased","mle","aspa"]
    midfixes = [midfix_a,midfix_c,midfix_b]
    M = len(midfixes)
    base = "output/deno/test/"
    gt = "data/benchmarks/"
    # W = 128
    W = 156
    # W = 128
    sigmas = [15,20,25,30,35]
    sigmas = [35]
    boxes = {
        "set5":{
            # "5_x1":[50,50]},
            "5_x1":[80,50]},

        # "set14":{#"9_x1":[225,225],
        #          # "2_x1":[100,420],
        #          # "2_x1":[120,450],
        #          # "8_x1":[120,120],
        #     # "11_x1":[100,250]
        #      },

        # "u100":{#"96_x1":[400,800],
        #         "5_x1":[600,500],
        # }
        "b100":{#"96_x1":[400,800],
                # "2_x1":[150,150],
                # "38_x1":[50,0],
                # "41_x1":[30,150+148],
            # "78_x1":[30,30],
            # "82_x1":[180,180],
        }
    }

    grids = []
    info = []
    for dname in boxes.keys():
        for iname,box in boxes[dname].items():
            imgs = []
            for sigma in sigmas:
                gt = get_boxes(get_gt(dname,iname),box,W)
                imgs.append(gt)
                for mname,midfix in zip(method_names,midfixes):
                    fn_m = Path(base)/str(sigma)/midfix/dname/("%s.png"%iname)
                    img_m = get_boxes(fn_m,box,W)
                    psnr_m = comp_psnrs(gt,img_m)
                    info.append({"dname":dname,"iname":iname,
                                  "sigma":sigma,"mname":mname,"psnrs":psnr_m})
                    imgs.append(img_m)
                # fn_b = Path(base)/str(sigma)/midfix_b/dname/("%s.png"%iname)
                # img_a,img_b = get_boxes(fn_a,fn_b,box,W)
                # imgs.append(img_a)
            # imgs = rearrange(th.stack(imgs),'(a b) c h w -> (b a) c h w',b=M+1)
            # grid = tv_utils.make_grid(imgs,nrow=len(sigmas))
            imgs = rearrange(th.stack(imgs),'(a b) c h w -> (a b) c h w',b=M+1)
            grid = tv_utils.make_grid(imgs,nrow=M+1)#len(sigmas))
            grids.append(grid)

    # -- get grid --
    # grid_a = tv_utils.make_grid(imgs_a,nrow=len(sigmas))
    # grid_b = tv_utils.make_grid(imgs_b,nrow=len(sigmas))
    # grid = tv_utils.make_grid(grids,nrow=M+1)

    for i,grid in enumerate(grids):
        H,W = grid.shape[-2:]
        grid_s = irz(grid,2*H,2*W)
        vid_io.save_image(grid_s,"output/figures/deno_examples/grid_%d"%i)
    # vid_io.save_image(grid,"output/figures/deno_examples/grid")

    info = pd.DataFrame(info)
    for di_name,di_df in info.groupby(["dname","iname"]):
        print(di_name)
        for mname,m_df in di_df.groupby(["mname"]):
            print(m_df)


if __name__ == "__main__":
    main()
