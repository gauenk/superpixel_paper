
import torch as th
from pathlib import Path
import pandas as pd
import numpy as np

from PIL import Image
from skimage.segmentation import mark_boundaries
from einops import rearrange,repeat
from torchvision.utils import draw_bounding_boxes,draw_segmentation_masks,save_image
# torchvision.utils.save_image

# -- nicer viz --
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

def get_sample():
    # iname = "35028"
    # iname = "302003"
    # iname = "126039"
    iname = "0048"
    # fn = Path("../BASS/pytorch_version_og/out_s123/csv/%s.csv" % iname)
    # fn = Path("../BASS/pytorch_version_og/out_img/csv/%s.csv" % iname)
    fn = Path("../BASS/pytorch_version_og/out_sp200_s123/csv/%s.csv" % iname)
    sp = np.genfromtxt(fn, delimiter=',').astype(np.int)
    fn = Path("../BASS/images/%s.jpg" % iname)
    if not(fn.exists()):
        fn = Path("../BASS/images/%s.png" % iname)
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

def viz_loc_box(img,box,col="red"):
    img = rearrange(img,'h w c -> c h w')
    img = th.from_numpy(img).type(th.uint8)
    box = th.tensor(box)
    img = draw_bounding_boxes(img,box[None,:],fill=True,colors=col)
    img = rearrange(img.numpy(),'c h w -> h w c')
    return img

def viz_loc_sp_seg(img,seg):
    img = rearrange(img,'h w c -> c h w')
    img = th.from_numpy(img).type(th.uint8)
    seg = th.tensor(seg)
    print(img.shape,seg.shape)
    img = draw_segmentation_masks(img, seg, alpha=0.8, colors="green")
    # img = draw_bounding_boxes(img,box[None,:],fill=True,colors="#FF00FF")
    img = rearrange(img.numpy(),'c h w -> h w c')
    return img


def draw_seg(img,imgSp):
    ncols = 1500
    # cm = pylab.get_cmap('gist_rainbow')
    cm = pylab.get_cmap('prism')
    def index2color(i):
        nmax = 200
        cm_i = ((1.*(i%nmax))/nmax) % 1
        color = [int(255.*i.item()) for i in cm(cm_i)]
        color = [color[0],color[1],color[2]]
        return tuple(color)

    # -- masks --
    masks = sp_to_mask(imgSp)
    img_ui = th.clamp(255*img,0,255).type(th.uint8)[0].cpu()
    color_map = [index2color(i%ncols) for i in range(ncols)]
    seg_result = draw_segmentation_masks(
        img_ui*0, masks.cpu(), alpha=0.3,
        # colors="blue",
        colors=color_map,
    )
    return seg_result.to(img.device)

def sp_to_mask(imgSp):
    # -- get masks --
    uniqs = imgSp.unique()
    masks = []
    for u in uniqs:
        masks.append(imgSp==u)
    masks = th.cat(masks)
    return masks


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
    elif img.ndim == 3 and s0 in [3,4]:
        img = rearrange(img,'h w c -> c h w')

    # print("img.shape: ",img.shape)
    is_t = th.is_tensor(img)
    if not is_t: img = th.from_numpy(img)
    img = TF.resize(img,(tH,tW),
                    interpolation=InterpolationMode.NEAREST)
    if not is_t: img = img.numpy()
    # print("[2] img.shape: ",img.shape)

    if ndim == 2:
        img = img[0]
    elif img.ndim == 3 and s0 == 3:
        img = rearrange(img,'c h w -> h w c')
    # if img.ndim > 2:
    #     img = img[:,:,:3]
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

def get_sp_list_from_square(sp,sp_id,hloc,wloc,W):
    # -- get box --
    sh,sw = hloc-W//2,wloc-W//2
    eh,ew = sh+W,sw+W
    sp_window = sp[sh:eh,sw:ew]
    uniq = np.unique(sp_window)
    return uniq.tolist()

def viz_alpha_window(img,sp,alphas,sp_id,hloc,wloc,W,K):

    # -- get box --
    sh,sw = hloc-W//2,wloc-W//2
    eh,ew = sh+W,sw+W
    sp_window = sp[sh:eh,sw:ew]
    img_window = img[sh:eh,sw:ew]
    hmid = img_window.shape[0]//2
    wmid = img_window.shape[1]//2

    # -- add alpha --
    img_alpha = np.ones(sp_window.shape)
    for (_sp_id,_alpha) in alphas:
        args = np.where(sp_window==_sp_id)
        if len(args[0]) == 0: continue
        # img_alpha[args] = _alpha
        if _alpha < 1e-10:
            for i in range(3):
                img_window[...,i][args] = 0
    img_alpha = img_alpha[:,:,None]
    img_alpha = (255.*img_alpha).astype(np.uint8)

    # -- viz box --
    # K = 31
    # print("info: ",hmid,wmid,K,W)
    sh,sw = hmid-K//2,wmid-K//2
    eh,ew = sh+K,sw+K
    kbox = [sh,sw,eh,ew]
    # print("kbox: ",kbox)
    img_window = viz_loc_box(img_window,kbox,"yellow")

    # -- append alpha --
    # print("img_window.shape: ",img_window.shape)
    img_window = np.concatenate([img_window,img_alpha],-1)
    # print("img_window.shape: ",img_window.shape)
    # print(img_window[hmid,wmid])
    # print(img_window)

    # -- center point --
    for i in range(2):
        for j in range(2):
            img_window[hmid+i,wmid+j] = [255,255,0,255]

    return img_window

def main():

    img,sp = get_sample()
    print(img.shape)
    print(sp.shape)

    W = 256
    Wviz = 48
    Kviz = 32
    # hi,wi = 115,130
    # hi,wi = 132,459
    hi,wi = 132,400
    sh,sw = hi-W//2,wi-W//2
    eh,ew = sh+W,sw+W
    print(img[sh:eh,sw:ew].shape)
    img = img[sh:eh,sw:ew]
    sp = sp[sh:eh,sw:ew]
    # img = irz(img[sh:eh,sw:ew],W*2,W*2)
    # sp = irz(sp[sh:eh,sw:ew],W*2,W*2)
    img_og = img
    sp_og = sp
    # sp_l = irz(sp,W*4,W*4)
    sp_l = irz(sp,W*1,W*1)
    # sp_id = sp[100,130]
    # sp_id = sp[140,95]
    # sp_id = sp[140,70]
    red_x= [90,128]
    sp_id = sp[red_x[0],red_x[1]]
    print(sp[125:145,70:80])
    # print(sp_id)
    # red_x= [280,148]
    # red_x= [265,153]
    # red_x= [265,153]
    print("red_x: ",red_x)
    # print("sp.shape: ",sp.shape)

    # -- viz image and seg --
    print("."*30)
    print("img.shape: ",img.shape)
    img_v = viz_loc_sp_seg(img.copy(),sp==sp_id)
    Image.fromarray(img_v).save("viz_region.png")
    # seg = mark_boundaries(img,sp,mode='subpixel',color=[1,0,0])
    seg = mark_boundaries(img,sp,mode='thick',color=[1,0,0])
    print("seg.shape,img.shape: ",seg.shape,img.shape)
    seg = np.clip(seg * 255,0,255.).astype(np.uint8)
    # seg_og = seg[::1,::1].copy()
    seg_og = seg.copy()
    print("seg.shape: ",seg.shape,sp_l.shape)
    # seg = viz_loc_sp_seg(seg.copy(),sp_l[:-1,:-1]==sp_id)
    seg = viz_loc_sp_seg(seg.copy(),sp_l==sp_id)
    print("seg.shape: ",seg.shape)
    print(seg[red_x[0],red_x[1]])
    # seg[red_x[0],red_x[1]] = [255,255,0]
    K = 4
    for i in range(K):
        for j in range(K):
            seg[red_x[0]+i,red_x[1]+j] = [255,255,0]

    sh,sw = -Kviz//2+red_x[0]+1,-Kviz//2+red_x[1]+1
    eh,ew = sh+Kviz,sw+Kviz
    kbox = [sw,sh,ew,eh]
    print("[viz_seg] kbox: ",kbox)
    seg = viz_loc_box(seg,kbox,col="yellow")
    Image.fromarray(seg).save("viz_seg.png")
    print("img.shape,seg_og.shape: ",img.shape,seg_og.shape)

    # -- viz a single sp --
    sp_img,hmid,wmid,sp_id = viz_sp(img.copy(),sp,sp_id)
    print(hmid,wmid)
    print(sp_img.shape)
    Image.fromarray(irz(sp_img,128,128)).save("viz_sp.png")

    # -- viz top-k sp --
    # K = 100
    # sp_img,hmid,wmid,sp_id = viz_sp(img.copy(),sp,sp_id)
    # sp_img = topk_sampling(sp_img.copy(),K)
    # Image.fromarray(irz(sp_img,128,128)).save("viz_ksp.png")


    # -- viz square with mask --
    # K = 35
    # hloc,wloc = 135,75
    # hloc,wloc = 2*135,2*75
    # hloc,wloc = 265,153
    hloc,wloc = red_x[0],red_x[1]
    # print(hloc,wloc)
    sp_list = get_sp_list_from_square(sp_l,sp_id,hloc,wloc,Wviz)
    print("sp_list: ",sp_list,sp_id)
    nat = [[s,1.] for s in sp_list]
    sna = [[s,1.*(sp_id==s)] for s in sp_list]
    # select = [252, 259, 266, 285, 290, 315]
    select = [67, 72, 80, 97, 103]
    print(sna)
    print("this: ",sp_list,sp_id)
    ssna = [[s,1.*(s in select)] for s in sp_list]
    print("."*10)
    print("img.shape,seg_og.shape: ",img.shape,seg_og.shape)
    nat_img = viz_alpha_window(seg_og.copy(),sp_l,nat,sp_id,hloc,wloc,Wviz,Kviz)
    sna_img = viz_alpha_window(seg_og.copy(),sp_l,sna,sp_id,hloc,wloc,Wviz,Kviz)
    ssna_img = viz_alpha_window(seg_og.copy(),sp_l,ssna,sp_id,hloc,wloc,Wviz,Kviz)
    nat_img = irz(nat_img,128,128)
    sna_img = irz(sna_img,128,128)
    ssna_img = irz(ssna_img,128,128)
    print(nat_img.shape)
    # nat_img = rearrange(irz(nat_img,128,128),'h w c -> c h w')
    # sna_img = rearrange(irz(sna_img,128,128),'h w c -> c h w')
    # ssna_img = rearrange(irz(ssna_img,128,128),'h w c -> c h w')
    save_image(th.from_numpy(nat_img)/255.,"viz_nat.png")
    save_image(th.from_numpy(sna_img)/255.,"viz_sna.png")
    save_image(th.from_numpy(ssna_img)/255.,"viz_ssna.png")
    # Image.fromarray(irz(ssna_img,128,128)).save("viz_ssna.png")
    # loc_img = viz_sp_square(img.copy(),sp,sp_id,hmid,wmid,ksize,vsize)


    # -- viz local region --
    ksize = 25
    vsize = 32
    loc_img = viz_sp_square(img.copy(),sp,sp_id,hmid,wmid,ksize,vsize)
    Image.fromarray(irz(loc_img,128,128)).save("viz_sp_loc.png")

    # ksize = 17
    # vsize = 55
    loc_img = viz_square(img.copy(),hmid,wmid,ksize,vsize)
    Image.fromarray(irz(loc_img,128,128)).save("viz_loc.png")


    # img_og = img
    # sp_og = sp
    # print("hi.")
    # img_seg = draw_seg(img_og,sp_og)
    # Image.fromarray(irz(img_seg,156,156)).save("viz_colors.png")

if __name__ == "__main__":
    main()

