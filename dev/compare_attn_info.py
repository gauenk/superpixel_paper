import pandas as pd
import torch as th
import numpy as np

from dev_basics.trte import bench
from superpixel_paper.deno_trte.train import load_model,extract_defaults,config_via_spa

import cache_io

def run_bench(cfg):

    cfg = extract_defaults(cfg)
    config_via_spa(cfg)
    model = load_model(cfg).cuda()
    _fwd = model.forward
    def fwd(vid): return _fwd(vid[:,0])[:,None]
    model.forward = fwd
    vshape = (cfg.batch_size,1,3,256,256)
    # vshape = (cfg.batch_size,1,3,512,512)
    summ = bench.summary_loaded(model,vshape,with_flows=False)
    # print(summ)
    # exit()
    return summ

def main():
    exp_fn = "exps/trte_deno/train_bench.cfg"
    cache_fn = ".cache_io_exps/trte_deno/bench/"
    exps,uuids = cache_io.train_stages.run(exp_fn,cache_fn,
                                           fast=False,update=True)
    # for e in exps: e.batch_size = 1
    # for e in exps: e.tag = "v0.01"
    results = cache_io.run_exps(exps,run_bench,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_deno/bench",
                                version="v1",skip_loop=False,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_deno/bench.pkl",
                                records_reload=False,use_wandb=False,
                                proj_name="superpixels_deno_bench")
    df = results.rename(columns={"spa_version":"spav","gen_sp_type":"gsp",
                                 "timer_fwd_nograd":"t_fwd_ng",
                                 "timer_fwd":"t_fwd",
                                 "timer_bwd":"t_bwd",
                                 "trainable_params":"params",
                                 "learn_attn_scale":"las"})
    df['params'] = df['params']*10**3
    df['a_params'] = df['params'] - .195
    fields0 = ["spav","gsp","las","params","a_params","t_fwd","t_bwd","alloc_fwd","alloc_bwd","seed"]
    df = df[fields0]
    fields = ["spav","gsp","las","params","a_params",
              "t_fwd","t_bwd","alloc_fwd","alloc_bwd"]

    copy_fields = ["spav","gsp","las","params","a_params"]
    ave_fields = ["t_fwd","t_bwd","alloc_fwd","alloc_bwd"]
    df = df.groupby(copy_fields).agg({f:'mean' for f in ave_fields})
    df['t_fwd'] = df['t_fwd']*10**3
    df['t_bwd'] = df['t_bwd']*10**3
    print(df)

    df = df.reset_index(drop=False)
    print(df)
    # -- format --
    fields = ave_fields
    order_las = [True,True,False,False,False,True,False]
    order_spav = ["ssna","ssna","ssna","ssna","sna","nat","nat"]
    order_gsp = ["ssn","modulated","ssn","modulated","default","none","none"]
    for f in fields:
        # df[['spav','las',f]]
        spav = df['spav'].to_numpy()
        las = df['las'].to_numpy()
        gsp = df['gsp'].to_numpy()
        finfo =  df[f].to_numpy()
        msg = ""
        for _las,_spav,_gsp in zip(order_las,order_spav,order_gsp):
            bool0 = np.logical_and(_las == las,_spav==spav)
            if _gsp != "none":
                bool1 = np.logical_and(_gsp == gsp,bool0)
            else:
                bool1 = bool0
            idx = np.where(bool1)[0]
            fmt = "%2.2f" % finfo[idx].item()
            msg += " " + str(fmt) + " &"
        print(msg)


    # for field,gdf in df.groupby(fields):
    #     print(fields)
    #     print(gdf)
    #     # df = df[fields]
    #     # print(df)

if __name__ == "__main__":
    main()
