
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
    exps,uuids = cache_io.train_stages.run("exps/trte_sr/train.cfg",
                                           ".cache_io_exps/trte_sr/train/",
                                           update=True)
    print("Num Exps: ",len(exps))
    print(uuids)
    # print(exps)
    # for e in exps:
    #     print(e.spa_version,e.topk)
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
