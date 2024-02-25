
"""

Training

"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- testing --
from superpixel_paper.deno_trte import train

# -- caching results --
import cache_io


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get experiments --
    def clear_fxn(num,cfg): return False
    # exp_fn = "exps/trte_deno/train_snet.cfg"
    # exp_fn = "exps/trte_deno/train.cfg"
    # exp_fn_list = ["exps/trte_deno/train_table.cfg"]
    exp_fn_list = [
        "exps/trte_deno/train_ksize.cfg",
        "exps/trte_deno/train_nsp.cfg",
        "exps/trte_deno/train_snfts.cfg",
        "exps/trte_deno/train_att_temp.cfg",
        "exps/trte_deno/train_att_temp_lrn.cfg",
    ]
    exps,uuids = [],[]
    for exp_fn in exp_fn_list:
        _exps,_uuids = cache_io.train_stages.run(exp_fn,
                                                 ".cache_io_exps/trte_deno/train/",
                                                 update=True)
        exps += _exps
        uuids += _uuids
    print("Num Exps: ",len(exps))
    print(uuids)
    # exps = list(reversed(exps))
    # uuids = list(reversed(uuids))
    # print(exps)
    # for e in exps:
    #     print(e.spa_version,e.topk,e.sigma)
    # exit()
    # df =pd.DataFrame(exps)
    # df = df[["gen_sp_type","gen_sp_use_grad","ssn_nftrs","share_gen_sp",
    #  "use_state","use_pwd"]]
    # print(df)
    # exit()

    # -- run exps --
    results = cache_io.run_exps(exps,train.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_deno/train",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_deno/train.pkl",
                                records_reload=False,use_wandb=False,
                                proj_name="superpixels_deno_train")



if __name__ == "__main__":
    main()
