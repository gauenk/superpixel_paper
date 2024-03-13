
# -- basic --
import numpy as np
from scipy.io import loadmat
import torch as th

# -- dataset --
from superpixel_paper.sr_datas.utils import get_seg_dataset

# -- superpixel eval --
from superpixel_paper.utils import metrics,extract_defaults
from superpixel_paper.utils.connected import connected_sp

# -- caching results --
import cache_io


def load_model(cfg):
    pass

def get_superpixel(cfg,img,model):
    if not(model is None):
        return get_superpixel_from_model(cfg,model,img)
    else:
        return get_superpixel_from_file(cfg,img)

def get_superpixel_from_model(cfg,model,img):
    pass

def get_superpixel_from_file(cfg,img):
    pass

def eval_bsd_500(cfg):

    # -- init experiment --
    defs = {"data_path":"./data/sr/"}
    cfg = extract_defaults(cfg,defs)
    device = "cuda"

    # -- dataset --
    dataloader = get_seg_dataset(cfg,load_test=True)

    # -- load (optional) model --
    model = load_model(cfg)

    # -- init info --
    ifields = ["asa","br","bp","nsp"]
    info = edict()
    for f in ifields: info[f] = []

    # -- each sample --
    for img,seg in dataloader:

        # -- unpack --
        img, seg = img.to(device)/255., seg.cpu().numpy()

        # -- get superpixel --
        spix = get_superpixel(cfg,img,model)
        spix = connected_sp(spix,0.5,3)

        # -- eval & collect info --
        iinfo = edict()
        for f in ifields: iinfo[f] = []
        iinfo.asa = metrics.compute_asa(spix,seg)
        iinfo.br = metrics.compute_br(spix,seg,r=1)
        iinfo.bp = metrics.compute_bp(spix,seg,r=1)
        iinfo.nsp = int(spix.max()+(spix.min()==0))
        for f in ifields: info[f].append(iinfo[f])

    return info

def main():

    methods = ["ssn","slic","ssna-deno-s10",
               "ssna-deno-s20","ssna-m-deno-s10","ssna-m-deno-s20"]
    results = cache_io.run_exps(exps,eval_bsd_500,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_deno/train",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_deno/train.pkl",
                                records_reload=False,use_wandb=False,
                                proj_name="superpixels_deno_train")

if __name__ == "__main__":
    main()
