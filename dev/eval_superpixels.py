
# -- basic --
import glob,os

from scipy.io import loadmat
from skimage.segmentation import mark_boundaries

import torch
import torch as th
import numpy as np
from torchvision.utils import save_image

from pathlib import Path
from einops import rearrange
from easydict import EasyDict as edict

# -- dataset --
from superpixel_paper.sr_datas.utils import get_seg_dataset

# -- superpixel eval --
from superpixel_paper.utils import metrics,extract_defaults
from superpixel_paper.utils.connected import connected_sp
from dev_basics.utils.metrics import compute_psnrs,compute_ssims

# -- caching results --
import cache_io

SAVE_ROOT = Path("output/eval_superpixels/")

def load_model(cfg):
    if cfg.method in ["ssn","ssn-pix","ssn-seg"]:

        # -- load config --
        cache = cache_io.ExpCache(".cache_io/trte_ssn/train","v1")
        _cfg = cache.get_config_from_uuid(cfg.model_uuid)

        # -- load model --
        from superpixel_paper.ssn_trte.train import load_model
        model = load_model(_cfg)

        # -- load weights --
        _cfg.log_path = "output/deno/train/"
        ckpt_path = Path(_cfg.log_path) / "checkpoints" / cfg.model_uuid
        chkpt_files = glob.glob(os.path.join(ckpt_path, "*.ckpt"))
        chkpt = torch.load(chkpt_files[-1])
        N=len("module.")
        state_dict = {k[N:]:v for k,v in chkpt['model_state_dict'].items()}
        model.load_state_dict(state_dict)
        model = model.cuda()

        return model
    elif "ssna" in cfg.method:

        # -- load config --
        cache = cache_io.ExpCache(".cache_io/trte_deno/train","v1")
        _cfg = cache.get_config_from_uuid(cfg.model_uuid)
        uuid = cfg.model_uuid
        assert (_cfg != -1),f"Invalid uuid. Check original training script. [{uuid}]"

        # -- load config --
        from superpixel_paper.deno_trte.train import load_model,extract_defaults
        from superpixel_paper.spa_config import config_via_spa
        _cfg = extract_defaults(_cfg)
        config_via_spa(_cfg)
        model = load_model(_cfg)

        # -- load weights --
        _cfg.log_path = "output/deno/train/"
        ckpt_path = Path(_cfg.log_path) / "checkpoints" / cfg.model_uuid
        chkpt_files = glob.glob(os.path.join(ckpt_path, "*.ckpt"))
        chkpt = torch.load(chkpt_files[-1])
        if _cfg.use_dataparallel: N=len("module.")
        else: N=0
        state_dict = {k[N:]:v for k,v in chkpt['model_state_dict'].items()}
        model.load_state_dict(state_dict)
        model = model.cuda()

        return model
    else:
        return

def get_superpixel(cfg,img,model,name):
    if cfg.method in ["ssn","ssn-pix","ssn-seg"]:
        return get_superpixels_from_ssn(cfg,model,img)
    elif "sna" in cfg.method:
        return get_superpixels_from_sna(cfg,model,img)
    # elif cfg.method == "sna-m":
    #     return get_superpixels_from_ssna(cfg,model,img)
    elif cfg.method == "slic":
        return get_superpixels_from_slic(cfg,img)
    elif cfg.method == "bass":
        return get_superpixels_from_bass(cfg,img,name)
    else:
        raise ValueError(f"Uknown superpixel method [{cfg.method}]")

def get_superpixels_from_slic(cfg,img):
    from skimage.segmentation import slic
    img = img.cpu().numpy()
    img = rearrange(img[0],'c h w -> h w c')
    H,W = img.shape[:2]
    Nsp = (H/cfg.S)*(W/cfg.S)
    spix = slic(img, n_segments=Nsp, compactness=10, sigma=1, start_label=1)
    return spix,None,None

def get_superpixels_from_ssn(cfg,model,img):
    img = img.cuda()

    # pad = torch.nn.functional.pad
    # img = pad(img,(1,2,1,2))
    B,_,Hp,Wp = img.shape
    assert B == 1
    sims = model(img)
    spix = sims.argmax(1)
    spix = spix.reshape(B,Hp,Wp)[0]
    # spix = spix[0,1:-2,1:-2]
    # sH,sW = Hp//cfg.S,Wp//cfg.S
    # sims = sims.reshape(B,sH,sW,Hp,Wp)
    # sims = sims[0]

    return spix,sims.detach(),None

def get_superpixels_from_sna(cfg,model,img):
    from superpixel_paper.models.sp_hooks import SsnaSuperpixelHooks
    sphooks = SsnaSuperpixelHooks(model)

    # pad = torch.nn.functional.pad
    # img = pad(img,(1,2,1,2))
    B,_,Hp,Wp = img.shape
    assert B == 1
    model = model.eval()
    # -- add noise if ssna --
    # if cfg.method == "ssna":
    #     img = img + (cfg.sigma/255.) * th.randn_like(img)
    noisy = img + (cfg.sigma/255.) * th.randn_like(img)
    # print(img)
    # print(noisy)
    # noisy = img
    # print("img.shape: ",img.shape,img.min(),img.max())
    deno = model(noisy*255)


    # deno = deno / 255.
    # check_psnr = compute_psnrs(img[:,None],deno[:,None],div=1.)[0].item()
    # print(check_psnr)

    sims = sphooks.spix[0]
    # print("sims.shape: ",sims.shape)
    # exit()
    sims = sims.reshape(Hp,Wp,-1)
    spix = th.argmax(sims,-1).long()
    # spix = spix[1:-2,1:-2]
    # print("num unique: ",len(th.unique(spix)))
    # print(spix[:10,:10])
    # print(spix[300:310,200:210])
    # exit()

    # sH,sW = Hp//cfg.S,Wp//cfg.S
    sims = rearrange(sims.view(Hp*Wp,-1),'npix nspix -> 1 nspix npix')


    return spix,sims.detach(),deno/255.

def get_superpixels_from_bass(cfg,img,name):
    # (10,1536),(12,1040),(14,748),(16,600),(18,442),(20,384),(24,260),(26,216)
    H,W = img.shape[-2:]
    nsp = (H//cfg.S)*(W//cfg.S)
    if nsp == 1040: nsp = 1024
    fn = Path(cfg.bass_path) / ("out_bsdtest_sp%d" % nsp) / "csv" / ("%s.csv"%name)
    spix = np.genfromtxt(fn, delimiter=",").astype(int)
    return spix,None,None

def sims_entropy(sims):
    # sims.shape = B, NumSuperpixels, NumPixels
    eps = 1e-6
    return th.mean(-sims*th.log(sims+eps)).item()

def eval_bsd_500(cfg):

    # -- init experiment --
    defs = {"data_path":"./data/sr/","data_augment":False,
            "patch_size":128,"data_repeat":1,"colors":3}
    cfg = extract_defaults(cfg,defs)
    device = "cuda"

    # -- dataset --
    dset,dataloader = get_seg_dataset(cfg,load_test=True)

    # -- load (optional) model --
    model = load_model(cfg)

    # -- init info --
    ifields = ["asa","br","bp","nsp_og","nsp","hw","name",
               "psnr","ssim","deno_psnr","deno_ssim","entropy"]
    info = edict()
    for f in ifields: info[f] = []

    # -- each sample --
    for ix,(img,seg) in enumerate(dataloader):

        # -- unpack --
        img, seg = img.to(device)/255., seg.cpu().numpy()[0]
        # print("img.shape: ",img.shape,seg.shape)
        img = img[:,:,:-1,:-1]
        seg = seg[:-1,:-1]
        # print("img.shape: ",img.shape,seg.shape)
        name = dset.names[ix]

        # -- get superpixel --
        spix,sims,deno = get_superpixel(cfg,img,model,name)
        if th.is_tensor(spix): spix = spix.cpu().numpy().astype(np.int64)
        else: spix = spix.astype(np.int64)
        spix_og = spix.copy()
        # print(spix.shape,sims.shape)
        if cfg.use_connected:
            cmin = cfg.connected_min
            cmax = cfg.connected_max
            spix = connected_sp(spix,cmin,cmax)
        # if cfg.method== "ssna":
        #     spix = connected_sp(spix,0.1,1.8)
        # elif ("sna" in cfg.method) or ("ssn" in cfg.method):
        #     spix = connected_sp(spix,0.5,3)
        # elif not(cfg.method in ["bass","ssna-m"]):
        #     spix = connected_sp(spix,0.5,3)
        # elif cfg.method == "slic":
        #     spix = connected_sp(spix,0.5,3)

        # -- save --
        if cfg.save:
            save_dir = SAVE_ROOT / ("s_%d"%cfg.S) / cfg.method
            if not save_dir.exists():
                save_dir.mkdir(parents=True)

            _img = img[0].cpu().numpy().transpose(1,2,0)
            viz = mark_boundaries(_img,spix,mode="subpixel")
            viz = th.from_numpy(viz.transpose(2,0,1))
            save_fn = save_dir / ("%s.jpg"%name)
            # print(save_fn)
            # print(viz.shape)
            # viz = viz[:,150:,400:]
            save_image(viz,save_fn)


            # pooled = sp_pool(img,spix,sims,cfg.S,cfg.method)
            # pooled = th.from_numpy(pooled.transpose(2,0,1))
            # save_fn = save_dir / ("%s_p.jpg"%name)
            # save_image(pooled,save_fn)

        deno_psnr = -1
        deno_ssim = -1
        if not(deno is None):
            # print(img.min(),img.max(),img.shape)
            # print(deno.min(),deno.max(),deno.shape)
            deno_psnr = compute_psnrs(img[:,None],deno[:,None],div=1.)[0].item()
            deno_ssim = compute_ssims(img[:,None],deno[:,None],div=1.)[0].item()
            print(deno_psnr)

        psnr = -1
        ssim = -1
        entropy = 0.
        if not(sims is None):
            _img = img[0].cpu().numpy().transpose(1,2,0)
            save_dir = SAVE_ROOT / ("s_%d"%cfg.S) / cfg.method
            entropy = sims_entropy(sims)
            smoothed = sp_smooth(_img,sims,cfg.S,cfg.method)
            smoothed = th.from_numpy(smoothed.transpose(2,0,1))
            if cfg.save:
                save_fn = save_dir / ("%s_s.jpg"%name)
                print(save_fn)
                save_image(smoothed,save_fn)
            hr,sr = _img,smoothed
            hr = rearrange(hr,'h w c -> c h w')
            hr = th.from_numpy(hr)
            psnr = compute_psnrs(hr[:,None],sr[:,None],div=1.)[0].item()
            ssim = compute_ssims(hr[:,None],sr[:,None],div=1.)[0].item()

        # -- eval & collect info --
        iinfo = edict()
        for f in ifields: iinfo[f] = []
        iinfo.asa = metrics.compute_asa(spix,seg)
        iinfo.br = metrics.compute_br(spix,seg,r=1)
        iinfo.bp = metrics.compute_bp(spix,seg,r=1)
        iinfo.nsp = int(len(np.unique(spix)))
        iinfo.nsp_og = int(len(np.unique(spix_og)))
        iinfo.psnr = psnr
        iinfo.ssim = ssim
        iinfo.deno_psnr = deno_psnr
        iinfo.deno_ssim = deno_ssim
        iinfo.entropy = entropy
        iinfo.hw = img.shape[-2]*img.shape[-1]
        iinfo.name = name
        print(iinfo)
        for f in ifields: info[f].append(iinfo[f])
        if cfg.num_samples > 0 and ix >= cfg.num_samples:
            break

    # exit()
    return info

def sp_smooth(img,sims,S,method):


    # -- alloc [compact loss] --
    H,W,F = img.shape
    is_tensor = th.is_tensor(img)
    if not th.is_tensor(img):
        img = th.from_numpy(img).to(sims.device)

    # -- pad --
    # pad = torch.nn.functional.pad
    # img = pad(rearrange(img,'h w f -> f h w'),(1,2,1,2))
    # img = rearrange(img,'f h w -> h w f')
    Hp,Wp = img.shape[:2]
    img = img.reshape(-1,F)

    # -- allocate --
    smoothed = th.zeros_like(img)
    # print("sims.shape: ",sims.shape)

    # -- normalize across #sp for each pixel --
    # sims.shape = B, NumSuperpixels, NumPixels
    sims_nmz = sims.transpose(-1,-2)
    sims_nmz = sims_nmz / sims_nmz.sum(-2,keepdim=True)

    # -- compute "superpixel loss" --
    for fi in range(F):
        img_fi = img[:,fi]
        img_fi = img_fi[None,:,None]
        # print("sims_nmz.shape: ",sims_nmz.shape)
        # print("sims.shape: ",sims.shape)
        # print("img_fi.shape: ",img_fi.shape)
        tmp = sims @ img_fi
        # print("tmp.shape: ",tmp.shape)
        tmp = sims_nmz @ (sims @ img_fi)
        # print("tmp.shape: ",tmp.shape)
        smoothed[:,fi] = (sims_nmz @ (sims @ img_fi))[0,:,0]
        # sp_loss = th.mean((labels - labels_sp)**2)

    # -- post proc --
    smoothed = smoothed.reshape(Hp,Wp,F)
    # smoothed = smoothed[1:-2,1:-2]
    if not is_tensor:
        smoothed = smoothed.cpu().numpy()

    return smoothed

    # print(sims.shape)
    # H,W,F = img.shape
    # sH,sW = (H+1)//S,(W+2)//S # add one for padding

def sp_pool(img,spix,sims,S,method):
    print(img.shape)
    H,W,F = img.shape
    if method in ["ssn","sna"]:
        sH,sW = (H+1)//S,(W+2)//S # add one for padding
    else:
        sH,sW = H//S,W//S # no padding needed

    is_tensor = th.is_tensor(img)
    if not th.is_tensor(img):
        img = th.from_numpy(img)
        spix = th.from_numpy(spix)

    img = img.reshape(-1,F)
    spix = spix.ravel()

    N = len(th.unique(spix))
    print(N,sH,sW)
    assert N <= (sH*sW)

    # -- normalization --
    counts = th.zeros((sH*sW),device=spix.device)
    ones = th.ones_like(img[:,0])
    counts = counts.scatter_add_(0,spix,ones)

    # -- pooled --
    pooled = th.zeros((sH*sW,F),device=spix.device)
    for fi in range(F):
        pooled[:,fi] = pooled[:,fi].scatter_add_(0,spix,img[:,fi])

    # -- exec normz --
    pooled = pooled/counts[:,None]

    # -- post proc --
    pooled = pooled.reshape(sH,sW,F)
    if not is_tensor:
        pooled = pooled.cpu().numpy()

    return pooled

def main():

    # -- base --
    print("PID: ",os.getpid())
    base = {"save":True,"num_samples":10,"global_tag":"v0.13"}

    # -- ssn_exps --
    tr_fn = "exps/trte_ssn/train.cfg"
    tr_fn = "exps/trte_ssn/train_again.cfg"
    ssn_exps,ssn_uuids = cache_io.train_stages.run(tr_fn,".cache_io_exps/trte_ssn/train/",
                                                   update=True)
    for u,e in zip(ssn_uuids,ssn_exps):
        e.model_uuid = u
        e.S = e.stoken_size
        e.use_connected = True
        e.connected_min = 0.1
        e.connected_max = 2
        e.ssn_target = e.target
        e.method = "ssn-" + e.ssn_target
        e.tag = "v0.1"

    # -- deno_exps --
    # tr_fn = "exps/trte_deno/train_nsp.cfg"
    #
    tr_fn = "exps/trte_deno/train_ssn.cfg"
    te_fn = "exps/trte_deno/test_sp_eval.cfg"
    #
    tr_fn = "exps/trte_deno/train_ssn_again.cfg"
    te_fn = "exps/trte_deno/test_sp_eval_again.cfg"
    #
    tr_exp = cache_io.fill_test_shell(tr_fn,te_fn)
    # print(tr_exp['train_grid']['mesh0'])
    # exit()
    read_test = cache_io.read_test_config.run
    _exps = read_test(tr_exp,".cache_io_exps/trte_deno/test_sp",
                      reset=True,skip_dne=True)
    # print(_exps[0])
    # exit()
    pair = cache_io.get_uuids(_exps,".cache_io/trte_deno/test_sp01",
                              read=False,no_config_check=False)
    # deno_exps,deno_uuids = cache_io.train_stages.run(fn,".cache_io/trte_deno/train/",
    #                                                # ".cache_io_exps/trte_deno/train/",
    #                                                  fast=False,update=True)
    deno_exps = pair[0]
    deno_uuids = [p.tr_uuid for p in deno_exps]
    # print(deno_exps[0])
    # exit()
    for u,e in zip(deno_uuids,deno_exps):
        e.model_uuid = u
        e.method = e.spa_version
        if e.gen_sp_type == "modulated":
            e.method += "-m"
        if e.deno_loss_lamb > 1e-10:
            e.method += "-deno1"
        else:
            e.method += "-deno0"
        if e.ssn_loss_lamb > 1e-10:
            e.method += "-ssn1"
        else:
            e.method += "-ssn0"
        e.method += "-%s" % e.ssn_target
        e.method += "-s%d" % e.sigma
        print(e.method)
        e.S = e.stoken_size
        is_again = e.tag == "v0.11"
        # print(is_again)
        # exit()
        e.tag = "v2.133"
        if is_again:
            e.method += "_again"
        e.use_connected = True
        e.connected_min = 0.1
        e.connected_max = 2
        if 'uuid' in e:
            del e['uuid']

    # -- unpack strides --
    Sgrid = np.unique([e.S for e in deno_exps]).tolist()

    # -- slic/bass exps --
    slic_exps = [{"method":"slic","S":S} for S in Sgrid]
    slic_exps = [edict(s) for s in slic_exps]
    for e in slic_exps:
        e.use_connected = True
        e.connected_min = 0.5
        e.connected_max = 3
    bass_path = "/home/gauenk/Documents/packages/"
    bass_path = bass_path + "BASS/pytorch_version_og/"
    bass_exps = [{"method":"bass","S":S,"bass_path":bass_path} for S in Sgrid]
    bass_exps = [edict(b) for b in bass_exps]
    for e in bass_exps:
        e.use_connected = False
        e.connected_min = -1
        e.connected_max = -1
    # 12,14,16,18,24
    # (10,1536),(12,1040),(14,748),(16,600),(18,442),(20,384),(24,260),(26,216)
    # - 12
    # - 14
    # - 16
    # - 20
    # - 24

    exps = ssn_exps + deno_exps + slic_exps + bass_exps
    # exps = [ssn_exps[0],deno_exps[0],slic_exps[0],bass_exps[0]]
    print([len(s) for s in [ssn_exps,deno_exps,slic_exps]])
    # exps = [ssn_exps[1],deno_exps[1],slic_exps[0]]
    exps = [ssn_exps[1],deno_exps[0]]#,slic_exps[0]]
    exps = [deno_exps[0],deno_exps[1]]
    # exps = [deno_exps[1],deno_exps[0]]
    exps = deno_exps + [ssn_exps[1],]

    # -- append base --
    for e in exps:
        for k in base:
            if not(k in e): e[k] = base[k]

    results = cache_io.run_exps(exps,eval_bsd_500,
                                name=".cache_io/eval_superpixels_v0/",
                                version="v1",skip_loop=False,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/eval_superpixels/run.pkl",
                                records_reload=True,use_wandb=False,
                                proj_name="eval_superpixels")

    print(results[['method','ssn_target','nsp','S','asa','br','bp','deno_psnr']])
    results = results.fillna(value=-1)
    # metrics = edict({f:-1 for f in ['asa','br','bp','psnr','ssim','entropy']})
    metrics = edict({f:-1 for f in ['asa','br','psnr','ssim','deno_psnr','deno_ssim','nsp','nsp_og']})
    # metrics = edict({f:-1 for f in ['psnr','ssim','entropy']})
    for method,mdf in results.groupby(["method",'ssn_target',"S"]):
        print(method)
        for f in metrics: metrics[f] = "%2.4f" % mdf[f].mean().item()
        print(metrics)

if __name__ == "__main__":
    main()
