

from PIL import Image
import torch as th
import numpy as np
from einops import rearrange

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image

from superpixel_paper.models.sp_modules import get_snn_pool

def resize(img,size=256):
    img = TF.resize(img,(size,size),interpolation=InterpolationMode.NEAREST)
    return img

def main():

    fn = "output/figures/cute_chick_cat.png"
    img = Image.open(fn).convert("RGB")
    img = th.tensor(np.array(img))/255.
    img = rearrange(img,'h w c -> c h w')
    img_og = img.clone()
    save_image(resize(img),"output/figures/slic_cat.png")

    S = 14
    p = 3
    F,H,W = img.shape
    sH,sW = H//S,W//S
    for i in range(sH):
        for j in range(sW):
            for pi in range(p):
                for pj in range(p):
                    img[:,S*i-p//2+pi,S*j-p//2+pj] = 0
                    img[0,S*i-p//2+pi,S*j-p//2+pj] = 1
    save_image(img,"output/figures/slic_cat_dotted.png")


    tH,tW = 256,256
    img = TF.center_crop(img[:,:-112,56:],(32,32))
    save_image(resize(img),"output/figures/slic_cat_zoom.png")

    pooled = get_snn_pool(img_og[None,:])[0]
    save_image(resize(pooled),"output/figures/pooled.png")

    img_cc = TF.center_crop(img_og[:,:-110,55:],(96,96))
    save_image(resize(img_cc),"output/figures/slic_cat_zoom_og.png")

    print(img_og.shape,pooled.shape)
    cc = 1
    # _viz = pooled[:,16-cc:16+cc,16-cc:16+cc]
    sh,sw = 13,18
    _viz = pooled[:,sh-cc:sh+cc+1,sw-cc:sw+cc+1]
    print(_viz.shape)
    save_image(resize(_viz),"output/figures/pooled_zoom.png")


if __name__ == "__main__":
    main()
