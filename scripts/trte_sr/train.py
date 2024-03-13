
"""

Training

"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- testing --
from superpixel_paper.sr_trte import train

# -- caching results --
import cache_io


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get experiments --
    def clear_fxn(num,cfg): return False
    # exp_fn = "exps/trte_sr/train_snet.cfg"
    # exp_fn = "exps/trte_sr/train.cfg"
    # exp_fn_list = ["exps/trte_sr/train_table.cfg"]
    exp_fn_list = [
        # "exps/trte_sr/train_table.cfg",
        "exps/trte_sr/train_att_temp_lrn.cfg",
        # "exps/trte_sr/train_snfts.cfg",
        # "exps/trte_sr/train_ksize.cfg",
        # "exps/trte_sr/train_nsp.cfg",
        # "exps/trte_sr/train_att_temp.cfg",
        # "exps/trte_sr/train_bare.cfg",
    ]
    exps,uuids = [],[]
    for exp_fn in exp_fn_list:
        _exps,_uuids = cache_io.train_stages.run(exp_fn,
                                                 ".cache_io_exps/trte_sr/train/",
                                                 update=True)
        # print(_exps[0]['batch_size'])
        # print(_exps[0]['batch_size_tr'])
        exps += _exps
        uuids += _uuids
    print("Num Exps: ",len(exps))
    print(uuids)

    # exps = list(reversed(exps))
    # uuids = list(reversed(uuids))
    # print(exps)
    # for e in exps:
    #     print(e.upscale)
    #     # print(e.spa_version,e.topk)
    # exit()

    # -- run exps --
    results = cache_io.run_exps(exps,train.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_sr/train",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_sr/train.pkl",
                                records_reload=False,use_wandb=False,
                                proj_name="superpixels_sr_train")



if __name__ == "__main__":
    main()
