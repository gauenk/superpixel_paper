
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
    exps = list(reversed(exps))
    uuids = list(reversed(uuids))
    print("Num Exps: ",len(exps))
    print(uuids)
    # for e in exps:
    #     print(e.upscale)

    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_sr/test",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_sr/test.pkl",
                                records_reload=True,use_wandb=False,
                                proj_name="superpixels_sr_test")

    results = results.rename(columns={"affinity_softmax":"asm"})
    results = results.fillna(value=-1)
    print(results[['spa_version','gen_sp_type']])
    print(results['spa_version'].unique())
    print(results['upscale'].unique())
    for sigma,sig_df in results.groupby("upscale"):
        print("-"*20)
        print("upscale: ",sigma)
        print("-"*20)
        # for dname,ddf in sig_df.groupby("dname"):
        #     print("dname: ",dname)
        #     # print(ddf.columns)
        for spa_version,spa_df in sig_df.groupby(["spa_version","gen_sp_type","nsa_mask_labels","share_gen_sp","conv_ksize","use_ffn"]):
            # print(spa_version)
            # print(spa_df['psnrs'])
            # exit()
            print("SPA: ",spa_version,"%2.2f,%0.3f" %
                  (spa_df['psnrs'].mean(),spa_df['ssims'].mean()))

    # # print(results.columns)
    # for spa,sdf in results.groupby("spa_version"):
    #     print("SPA: ",spa)
    #     # print(sdf[['dname','name','topk','psnrs','ssims']])
    #     for dname,ddf in sdf.groupby("dname"):
    #         # if dname != "set5": continue
    #         print("[%s]: %2.2f,%0.3f" % (dname,ddf['psnrs'].mean(),ddf['ssims'].mean()))
    #         # print(ddf[['name','topk','psnrs','ssims']])


if __name__ == "__main__":
    main()
