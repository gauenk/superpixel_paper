"""

Compare the impact of train/test using exact/refineimate methods


"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- testing --
from superpixel_paper.ssn_trte import train

# -- caching results --
import cache_io


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get experiments --
    def clear_fxn(num,cfg): return False
    fn = "exps/trte_ssn/train.cfg"
    fn = "exps/trte_ssn/train_again.cfg"
    exps,uuids = cache_io.train_stages.run(fn,".cache_io_exps/trte_ssn/train/",
                                           update=True)
    print("Num Exps: ",len(exps))
    print(uuids)

    # -- run exps --
    results = cache_io.run_exps(exps,train.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_ssn/train",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_ssn/train.pkl",
                                records_reload=False,use_wandb=False,
                                proj_name="ssn_train")



if __name__ == "__main__":
    main()
