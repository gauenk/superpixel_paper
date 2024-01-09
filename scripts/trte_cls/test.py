
"""

Training

"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- testing --
from superpixel_paper.cls_trte import test

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
    exps = read_test("exps/trte_cls/test.cfg",
                     ".cache_io_exps/trte_cls/test",reset=refresh,skip_dne=refresh)
    exps,uuids = cache_io.get_uuids(exps,".cache_io/trte_cls/test",
                                    read=not(refresh),no_config_check=False)
    print("Num Exps: ",len(exps))
    print(uuids)

    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_cls/test",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_cls/test.pkl",
                                records_reload=True,use_wandb=False,
                                proj_name="superpixels_cls_test")

    for sigma,sig_df in results.groupby("sigma"):
        print("sigma: ",sigma)
        for spa,sdf in sig_df.groupby(["spa_version","topk"]):
            print("SPA: ",spa)
            # print(sdf[['dname','name','topk','psnrs','ssims']])
            for dname,ddf in sdf.groupby("dname"):
                # if dname != "set5": continue
                print("[%s]: %2.2f,%0.3f" % (dname,ddf['psnrs'].mean(),ddf['ssims'].mean()))

    # for sigma,sig_df in results.groupby("sigma"):
    #     print("sigma: ",sigma)
    #     for f0,f0_df in sig_df.groupby(["spa2_kweight","spa2_normz"]):
    #         print("f0: ",f0)
    #         for dname,ddf in f0_df.groupby("dname"):
    #             print("[%s]: %2.2f,%0.3f" % (dname,ddf['psnrs'].mean(),ddf['ssims'].mean()))

if __name__ == "__main__":
    main()
