
"""

Training

"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- testing --
from superpixel_paper.sr_trte import test

# -- caching results --
import cache_io


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get/run experiments --
    refresh = True
    def clear_fxn(num,cfg): return False
    read_test = cache_io.read_test_config.run
    exps = read_test("exps/trte_sr/test.cfg",
                     ".cache_io_exps/trte_sr/test",reset=refresh,skip_dne=refresh)
    exps,uuids = cache_io.get_uuids(exps,".cache_io/trte_sr/test",
                                    read=not(refresh),no_config_check=False)
    print("Num Exps: ",len(exps))
    print(uuids)

    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_sr/test",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_sr/test.pkl",
                                records_reload=True,use_wandb=False,
                                proj_name="superpixels_sr_test")

    # print(results.columns)
    for spa,sdf in results.groupby("spa_version"):
        print("SPA: ",spa)
        # print(sdf[['dname','name','topk','psnrs','ssims']])
        for dname,ddf in sdf.groupby("dname"):
            # if dname != "set5": continue
            print("[%s]: %2.2f,%0.3f" % (dname,ddf['psnrs'].mean(),ddf['ssims'].mean()))
            # print(ddf[['name','topk','psnrs','ssims']])


if __name__ == "__main__":
    main()
