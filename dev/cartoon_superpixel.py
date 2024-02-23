
# -- basic --
import numpy as np
import torch as th
from einops import rearrange

# -- nicer viz --
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision import utils as tv_utils
from torchvision.utils import save_image,make_grid
from skimage.segmentation import mark_boundaries


def irz(img,tH,tW):
    is_t = th.is_tensor(img)
    if not is_t: img = th.from_numpy(img)
    img = TF.resize(img,(tH,tW),
                    interpolation=InterpolationMode.NEAREST)
    if not is_t: img = img.numpy()
    return img

# def draw_boundary(img,bnd_col):
#     inds = []
#     F,H,W = img.shape
#     prev = img[:,0,0]
#     for hi in range(H):
#         for wi in range(W):
#             curr = img[:,hi,wi]
#             if th.all(prev == curr) or (wi == 0):
#                 if wi == 0: prev = curr
#                 continue
#             else:
#                 inds.append([hi,wi])
#             prev = curr
#                 # img[:,hi,wi] = bnd_col

#     for wi in range(W):
#         for hi in range(H):
#             curr = img[:,hi,wi]
#             if th.all(prev == curr) or (hi == 0):
#                 if hi == 0: prev = curr
#                 continue
#             else:
#                 inds.append([hi,wi])
#             prev = curr
#     for (hi,wi) in inds:
#         img[:,hi,wi] = bnd_col

def main():

    R = 30
    H,W = 9,9
    img = th.ones((3,H,W))
    col = th.tensor([31, 119, 180])/255.
    C = len(col)

    # -- red --
    img[0,:,:] = 255/255.
    img[1,...] = 127/255.
    img[2,...] = 14/255.

    # -- why not this? --
    nW = W//2
    img[:,:,:nW] = col[:,None,None].expand_as(img[:,:,:nW])
    img[:,H//2,W//2] = col

    # -- add alpha mask--
    # tH,tW = 32,32
    # img = irz(img,tH,tW)
    # bnd_col = th.tensor([44, 160, 44])/255.
    # draw_boundary(img,bnd_col)
    tH,tW = 128,128
    img = irz(img,tH,tW)


    # -- mark boundaries --
    labels = th.all(img==col[:,None,None].expand_as(img),0).long()
    # print(labels)
    # print(labels.shape)
    img = img.permute(1,2,0)*255.
    # print(img.shape,labels.shape)
    seg = mark_boundaries(img.numpy(),labels.numpy(),mode="thick",color=(255,255,0))
    # print(seg)
    # print(seg.shape)
    seg = th.tensor(seg).permute(2,0,1)/255.
    print(seg.shape)

    # -- save --
    seg0 = seg.clone()
    save_image(seg,"output/figures/cartoon_superpixels_0.png")
    col = seg[:,0,0]
    args = th.where(th.all(seg == seg[:,-1:,-1:],0))
    for i in range(3): seg[i][args] = seg[i,0,0]
    seg1 = seg.clone()
    save_image(seg,"output/figures/cartoon_superpixels_1.png")


    # -- NA windows --
    cH,cW = seg.shape[-2]//2,seg.shape[-1]//2
    sH,sW = cH - R//2,cW - R//2
    eH,eW = sH+R,sW+R
    save_image(irz(seg0[:,sH:eH,sW:eW],tH,tW),"output/figures/na_0.png")
    save_image(irz(seg1[:,sH:eH,sW:eW],tH,tW),"output/figures/na_1.png")

    # -- SNA windows --
    cH,cW = seg.shape[-2]//2,seg.shape[-1]//2
    sH,sW = cH - R//2,cW - R//2
    eH,eW = sH+R,sW+R
    sna0 = seg0[:,sH:eH,sW:eW].clone()
    sna1 = seg1[:,sH:eH,sW:eW].clone()
    args = th.where(th.all(sna0 == sna0[:,-1:,-1:],0))
    for i in range(3): sna0[i][args] = 0
    args = th.where(th.all(sna0 == sna0[:,-1:,-1:],0))
    for i in range(3): sna1[i][args] = 0
    save_image(irz(sna0,tH,tW),"output/figures/sna_0.png")
    save_image(irz(sna1,tH,tW),"output/figures/sna_1.png")


if __name__ == "__main__":
    main()
