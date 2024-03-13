"""

A script to highlight regions from denoised examples

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

SAVE_DIR = Path("./output/deno_examples/")

def get_regions(vid_name):
    if vid_name == "3":
        regions = []
        regions.append('70_160_64')
        # regions.append('0_95_156')
    elif vid_name == "11":
        regions = []
        regions.append('80_200_48')
        # regions.append('70_190_64')
        # regions.append('30_150_128')
    elif vid_name == "19":
        regions = []
        regions.append('0_320_156')
    elif vid_name == "22":
        regions = []
        regions.append('30_180_128')
    elif vid_name == "40":
        regions = []
        # regions.append('30_300_64')
        regions.append('30_310_48')
        # regions.append('30_250_128')
    elif vid_name == "51":
        regions = []
        # regions.append('140_110_32')
        regions.append('130_150_32')
    elif vid_name == "57":
        regions = []
        regions.append('60_210_32')
        # regions.append('60_210_64')
    elif vid_name == "66":
        regions = []
        regions.append('100_150_200')
    else:
        print(f"Please select a real region for {vid_name}!")
        regions = []
        regions.append('0_0_350')
    return regions

def vid_region(vid,region):
    h0,w0,rsize = [int(r) for r in region.split("_")]
    hslice = slice(h0,h0+rsize)
    wslice = slice(w0,w0+rsize)
    return vid[0,:,hslice,wslice]

# def highlight_mask(vid,region):
#     t,h0,w0,h1,w1 = [int(r) for r in region.split("_")]
#     hslice = slice(h0,h1)
#     wslice = slice(w0,w1)
#     mask = th.zeros_like(vid).bool()
#     mask[[t],:,hslice,wslice] = 1
#     return mask

def ensure_size(img,tH,tW):
    # img = TF.resize(img,(tH,tW),InterpolationMode.BILINEAR)
    img = TF.resize(img,(tH,tW),InterpolationMode.NEAREST)
    return img

def increase_brightness(img, value=30):

    is_t = th.is_tensor(img)
    if th.is_tensor(img):
        img = img.cpu().numpy()
        img = rearrange(img,'c h w -> h w c')
    # device = img.device
    # img = img.cpu().numpy()
    # img = rearrange(img,'c h w -> h w c')*1.
    img = img*1.
    img /= img.max()
    img *= 255
    img = np.clip(img,0,255).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # lim = 255 - value
    # v[v > lim] = 255
    # v[v <= lim] += value
    v = v*1.
    v -= v.min()
    v /= v.max()
    v *= 255
    v = v.astype(np.uint8)


    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

    # img = th.from_numpy(img).to(device)
    # img = rearrange(img,'h w c -> c h w')
    # img /= img.max()
    if is_t:
        img = th.from_numpy(img)
        img = rearrange(img,'h w c -> c h w')
        img = img/255.


    return img

def higlight_slice(vid,region,color,alpha=0.4):
    # -- highlight selected region --
    mask = highlight_mask(vid,region)
    t = mask.shape[0]
    for ti in range(t):
        vid[ti] = draw_segmentation_masks(vid[ti],mask[ti],alpha=alpha,colors=color)
    return vid

def get_vid_from_data(data,vid_name):
    groups = data.groups
    indices = [i for i,g in enumerate(groups) if vid_name in g]
    assert len(indices) == 1
    vid = data[indices[0]]['clean']
    return vid

def save_video(vid,root,fname):

    # -- format --
    if th.is_tensor(vid):
        vid = vid.cpu().numpy()
    if vid.dtype != np.uint8:
        if vid.max() < 200: # guess at normalization
            vid /= vid.max()
            vid *= 255
            vid = np.clip(vid,0.,255.)
        vid = vid.astype(np.uint8)

    # -- create root --
    if not root.exists():
        root.mkdir(parents=True)

    # -- save burst --
    vid = rearrange(vid,'t c h w -> t h w c')
    t = vid.shape[0]
    for ti in range(t):
        vid_t = increase_brightness(vid[ti])
        # vid_t = vid[ti]
        vid_t = Image.fromarray(vid_t)
        fname_t = fname + "_%d.png" % ti
        vid_t.save(str(root / fname_t))

# def save_regions(vid,regions,root,fname):
#     for r,region in enumerate(regions):
#         vid_r = vid_region(vid,region)
#         # print(fname+"_%d"%r,vid_r.shape)
#         # save_video(vid_r,root,fname+"_%d"%r)


# def save_highlight(vid,regions,root,fname):
#     vid = (vid).type(th.uint8)
#     colors = ["red","yellow","blue"]
#     for r,region in enumerate(regions):
#         if r == 0: continue
#         cidx = r % len(colors)
#         color = colors[cidx]
#         vid = higlight_slice(vid,region,color,alpha=0.4)
#     save_video(vid,root,fname)

def load_from_results(df):
    home = Path(df['home_path'].iloc[0])
    paths = df['deno_fns'].iloc[0][0]
    vid = []
    for path in paths:
        path_f = home / path
        img = Image.open(path_f).convert("RGB")
        img = np.array(img)
        img = rearrange(img,'h w c -> c h w')
        vid.append(img)
    vid = np.stack(vid)
    return vid

# def get_c

def save_examples(vids,root,regions,psnrs,vname,order=None):

    # # -- show zoomed regions on larger vid --
    # save_highlight(vids.clean,regions,root,"clean_highlight")
    # if order is None:
    #     order = list(vids.keys())

    # -- save zoomed regions --
    for rx,region in enumerate(regions):
        regs = []
        for vid_cat,vid in vids.items():
            print(vid_cat)
            reg = vid_region(vid,region)#,root,vid_cat)
            reg = ensure_size(reg,156,156)
            regs.append(reg)
        grid = make_grid(regs)/255.
        name = "%s_%d.png" % (vname,rx)
        fn = root/name
        print(fn)
        if vname in ["66"]:
            grid = increase_brightness(grid)
        save_image(grid,fn)
        return grid

def load_denoised(cfg):
    path_s = "./output/deno/test/%02d/%s/epoch=%d/%s/%s_x1.png"
    r_uuid = cfg.pretrained_path[:5]
    epoch = int(cfg.pretrained_path.split("=")[-1].split(".")[0])
    dname = cfg.eval_sets.lower()
    path_s = path_s % (int(cfg.sigma),r_uuid,epoch,dname,cfg.vid_name)
    img = np.array(Image.open(str(path_s)))
    img = th.tensor(img.transpose(2,0,1))
    return img

def prepare_vids(cfgs):

    # -- load data --
    vids = edict()
    cfg = cfgs['ssna-d']
    from superpixel_paper.sr_datas.utils import create_datasets
    from superpixel_paper.deno_trte.train import extract_defaults
    dcfg = extract_defaults(cfg)
    train_dataloader, valid_dataloaders = create_datasets(dcfg)
    ds = [val['data'] for val in valid_dataloaders if val['name'] == 'bsd68'][0]
    name = ds.names[int(cfg.vid_name)-1]
    clean = ds[int(cfg.vid_name)-1][1]
    noisy = clean + cfg.sigma*th.randn_like(clean)

    # -- format noisy --
    clean = (th.clamp(clean,0.,255.)).type(th.uint8)[None,:]
    noisy = (th.clamp(noisy,0.,255.)).type(th.uint8)[None,:]
    vids.noisy,vids.clean = noisy,clean

    # -- load denoised --
    # cfgs = {"ssna-d":cfg_ssnad,"ssna-m":cfg_ssnam,"sna":cfg_sna,"nat":cfg_nat}
    for _key,_cfg in cfgs.items():
        vids[_key] = load_denoised(_cfg).type(th.uint8)[None,:]

    # for key in vids:
    #     print(key,vids[key].shape)
    return vids

def get_psnrs(vids,regions):
    psnrs = edict({key:[] for key in vids})
    for rx,region in enumerate(regions):
        vid_c = vid_region(vids.clean,region)[None,:]
        for key in vids:
            vid_k = vid_region(vids[key],region)[None,:]
            psnrs[key].append(compute_psnrs(vid_c,vid_k,255).item())
    # psnrs.ssna = compute_psnrs(vids.clean,vids.ssna,255)
    # psnrs.sna = compute_psnrs(vids.clean,vids.sna,255)
    # psnrs.nat = compute_psnrs(vids.clean,vids.nat,255)
    return psnrs

# def prepare_psnrs(cfg_ours,cfg_orig,vids,regions,nframes,frame_start):

#     # -- endpoints --
#     if nframes > 0:
#         fs,fe = frame_start,frame_start + nframes
#     else:
#         fs,fe = 0,-1

#     # -- unpack "t" list --
#     print(regions)
#     tlist = [int(r.split("_")[0]) for r in regions]

#     # -- noisy psnrs --
#     npsnrs = compute_psnrs(vids.noisy,vids.clean,255)
#     print(cfg_ours['psnrs'])

#     # -- psnrs compact --
#     psnrs = edict()
#     psnrs.clean = [0. for t in tlist]
#     psnrs.noisy = [npsnrs[t] for t in tlist]
#     psnrs.deno_ours = [cfg_ours['psnrs'][0][t] for t in tlist]
#     psnrs.deno_orig = [cfg_orig['psnrs'][0][t] for t in tlist]
#     return psnrs

# def save_deltas(vids,root):
#     vids.deno_ours - vids.clean
#     vids.deno_orig - vids.clean

# def get_topk_regions(vids,K):
#     ours = th.abs(vids.clean - vids.deno_ours)
#     orig = th.abs(vids.clean - vids.deno_orig)
#     delta = orig - ours
#     print(delta.shape)

#     T,C,H,W = vids.clean.shape
#     windows = [64,128]
#     def get_grid(ws,N):
#         locs_h = th.linspace(ws//2,int(H-ws//2),N)
#         locs_w = th.linspace(ws//2,int(W-ws//2),N)
#         grid_y, grid_x = th.meshgrid(locs_h,locs_w)
#         grid = th.stack((grid_y, grid_x),2).long()  # H(x), W(y), 2
#         grid = grid.view(-1,2)
#         return grid

#     vals = []
#     regions = []
#     for t in range(T):
#         for ws in windows:
#             grid = get_grid(ws,8)
#             for loc in grid:
#                 print(t,ws,loc)
#                 sH,sW = max(loc[0]-ws//2,0),max(loc[1]-ws//2,0)
#                 eH,eW = sH+ws,sW+ws
#                 # val_d = th.mean(delta[t,:,sH:eH,sW:eW]).item()
#                 val_ours = th.mean(ours[t,:,sH:eH,sW:eW]).item()
#                 val_orig = th.mean(orig[t,:,sH:eH,sW:eW]).item()
#                 vals.append(val_orig/(val_ours+1e-8))
#                 reg_loc = "%d_%d_%d_%d_%d" % (t,sH,sW,eH,eW)
#                 regions.append(reg_loc)
#     vals = th.tensor(vals)
#     topk = th.topk(vals,K,largest=True)
#     regions = [regions[k] for k in topk.indices]
#     print(regions,topk.values)
#     return regions

def run(cfgs,order=None):

    # -- save info --
    cfg = cfgs['ssna-d']
    save_dir = Path("./output/deno_examples/%02d/" % cfg.sigma)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    root = SAVE_DIR

    # -- load videos --
    vids = prepare_vids(cfgs)

    # -- get region --
    vid_name = cfg['vid_name']
    regions = get_regions(vid_name)

    # -- load psnrs --
    # psnrs = prepare_psnrs(cfg,cfg_orig,vids,regions,nframes,frame_start)
    psnrs = get_psnrs(vids,regions)
    print(psnrs)

    # -- save region --
    grid = save_examples(vids,root,regions,psnrs,cfg.vid_name,order)

    # -- save delta --
    # save_deltas(vids,root)
    return grid,psnrs

def get_paired_deno(vids):

    # -- read exps --
    refresh = True
    tr_fn_list = ["exps/trte_deno/train_att_temp_lrn.cfg",
                  "exps/trte_deno/train_att_temp_lrn_sna.cfg",
                  "exps/trte_deno/train_att_temp_lrn_sna-m.cfg"]
    # te_fn = "exps/trte_deno/test_shell.cfg"
    te_fn = "exps/trte_deno/test_shell_bsd68.cfg"

    exps = []
    for tr_fn in tr_fn_list:
        print(tr_fn)
        tr_exp = cache_io.fill_test_shell(tr_fn,te_fn)
        read_test = cache_io.read_test_config.run
        _exps = read_test(tr_exp,".cache_io_exps/trte_deno/test",
                          reset=refresh,skip_dne=refresh)
        exps += _exps
        print(exps[-1])

    # -- load-up info --
    paired_cfgs = {}
    # spa_version = ["ssna","ssna","sna","nat"]
    # gen_sp_type = ["ssn","modulated","default","none"]
    # names = ["ssna-d","ssna-m","sna","nat"]
    # spa_version = ["ssna","ssna","nat"]
    # gen_sp_type = ["ssn","modulated","none"]
    # names = ["ssna-d","ssna-m","nat"]
    learn_attn_scale = [True,False]
    spa_version = ["ssna","nat"]
    # gen_sp_type = ["ssn","none"]
    gen_sp_type = ["modulated","none"]
    names = ["ssna-d","nat"]
    for vid in vids:
        paired_cfgs[vid] = {}
        for lrn_attn in learn_attn_scale:
            lrn_t = "lrn" if lrn_attn else "fixed"
            paired_cfgs[vid][lrn_t] = {}
            for ix,name in enumerate(names):
                for exp in exps:
                    lrn = exp['learn_attn_scale']
                    spa = exp['spa_version']
                    gen = exp['gen_sp_type']
                    # print(lrn,spa,gen)
                    _bool = (exp['learn_attn_scale'] == lrn_attn)
                    _bool = _bool and (exp['spa_version'] == spa_version[ix])
                    _bool = _bool and (exp['gen_sp_type'] == gen_sp_type[ix])
                    _bool = _bool and (exp['sigma'] == 20)
                    if _bool:
                        _exp = dcopy(exp)
                        _exp.vid_name = vid
                        _exp.upscale = 1
                        paired_cfgs[vid][lrn_t][name] = _exp
    return paired_cfgs
    # for rvrt in cfgs.rvrt:
    #     if rvrt.sigma != 50: continue
    #     name = rvrt.vid_name
    #     if not(name in vids): continue
    #     nlnet = [nlnet for nlnet in cfgs.nlnet if nlnet.vid_name == name]
    #     nlnet = [n for n in nlnet if n.sigma == 50][0]
    #     rvrt.uuid = cache.rvrt.get_uuid(rvrt)
    #     nlnet.uuid = cache.nlnet.get_uuid(nlnet)
    #     # print(rvrt.vid_name,rvrt.uuid,nlnet.uuid)
    #     print(rvrt.vid_name,rvrt.uuid,nlnet.uuid)
    #     paired_cfgs[name] = {"orig":rvrt,"ours":nlnet}
    #     # pp.pprint(cfg)
    #     # res = cache.load_exp(cfg) # load result
    #     # print(res.keys())
    #     # paired_cfgs[name][cls] = res
    #     # break
    #     # if ix > 5: break

    return paired_cfgs

def main():

    # -- create denos --
    vids = ["40"]
    # vids = ["11","57"]
    # vids = ["22"]
    # vids = ["19"]
    # vids = ["3","40","57","11","66","14"]
    # vids = ["11","40","57","3"]
    # vids = ["11","40","57","51"]
    vids = ["40","57","51"]
    learn_attn_scale = ['lrn','fixed']
    paired_cfgs = get_paired_deno(vids)
    df = []
    grids = []
    for vid_name in paired_cfgs.keys():
        # for lrn in learn_attn_scale:
            # print(list(paired_cfgs.keys()))
            # print(list(paired_cfgs[vid_name].keys()))
            # print(list(paired_cfgs[vid_name][lrn].keys()))
            # ssnad = paired_cfgs[vid_name][lrn]['ssna-d']
            # ssnam = paired_cfgs[vid_name][lrn]['ssna-m']
            # sna = paired_cfgs[vid_name][lrn]['sna']
            # nat = paired_cfgs[vid_name][lrn]['nat']
        exps = edict()
        exps['ssna-dm'] = paired_cfgs[vid_name]['lrn']['ssna-d']
        exps['ssna-d'] = paired_cfgs[vid_name]['fixed']['ssna-d']
        # exps['nat-d'] = paired_cfgs[vid_name]['lrn']['ssna-d']
        exps['nat-l'] = paired_cfgs[vid_name]['lrn']['nat']
        exps['nat'] = paired_cfgs[vid_name]['fixed']['nat']
        _grid,_psnrs = run(exps)
        grids.append(_grid)
        _psnrs = {k:p[0] for k,p in _psnrs.items()}
        _psnrs['name'] = vid_name
        df.append(_psnrs)

    # -- pandas --
    pd.set_option("display.precision", 2)
    df = pd.DataFrame(df)
    print(df)

    # -- save grids --
    grid = make_grid(grids,nrow=1,padding=0)
    fn = SAVE_DIR/("%s_%d.png" % ("grid",0))
    print(fn)
    print(grid.shape)
    save_image(grid,fn)


if __name__ == "__main__":
    main()
