
"""

Training

"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- testing --
from superpixel_paper.deno_trte import test

# -- caching results --
import cache_io


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)
    read_testing = False

    # -- get/run experiments --
    refresh = True and not(read_testing)
    def clear_fxn(num,cfg): return False
    read_test = cache_io.read_test_config.run
    # fn = "exps/trte_deno/test_early.cfg"
    # fn = "exps/trte_deno/test_snet.cfg"
    # fn = "exps/trte_deno/test.cfg"

    # -=-=-=-=-=-=-=-=-=-=-=-
    #
    # --  operator mode 1  --
    #
    # -=-=-=-=-=-=-=-=-=-=-=-

    train_fn_list = [
        # "exps/trte_deno/train_table.cfg",
        # "exps/trte_deno/train_ksize.cfg",
        # "exps/trte_deno/train_nsp.cfg",
        # "exps/trte_deno/train_snfts.cfg",
        # "exps/trte_deno/train_att_temp.cfg",
        "exps/trte_deno/train_att_temp_lrn.cfg",
    ]
    te_fn = "exps/trte_deno/test_shell.cfg"
    exps,uuids = [],[]
    for tr_fn in train_fn_list:
        tr_exp = cache_io.fill_test_shell(tr_fn,te_fn)
        _exps = read_test(tr_exp,".cache_io_exps/trte_deno/test",
                         reset=refresh,skip_dne=refresh)
        _exps,_uuids = cache_io.get_uuids(_exps,".cache_io/trte_deno/test",
                                          read=not(refresh),no_config_check=False)
        exps += _exps
        uuids += _uuids


    # -=-=-=-=-=-=-=-=-=-=-=-
    #
    # --  operator mode 0  --
    #
    # -=-=-=-=-=-=-=-=-=-=-=-

    # fn = "exps/trte_deno/test_table.cfg"
    # exps = read_test(fn,".cache_io_exps/trte_deno/test",reset=refresh,skip_dne=refresh)
    # exps,uuids = cache_io.get_uuids(exps,".cache_io/trte_deno/test",
    #                                 read=not(refresh),no_config_check=False)
    print("Num Exps: ",len(exps))
    print(uuids)

    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_deno/test",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_deno/test.pkl",
                                records_reload=True and not(read_testing),
                                use_wandb=False,proj_name="superpixels_deno_test")

    results = results.rename(columns={"affinity_softmax":"asm"})
    results = results.fillna(value=-1)
    print(results[['spa_version','gen_sp_type']])
    print(results['spa_version'].unique())
    for sigma,sig_df in results.groupby(["sigma","learn_attn_scale"]):
        print("-"*20)
        print("sigma: ",sigma)
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

            # fields = ["block_num","conv_ksize"]
        #     fields = ["use_state","use_pwd","ssn_nftrs"]

        #     spa_df = spa_df.sort_values(fields)
        #     for spa,sdf in spa_df.groupby(fields+["tr_uuid",],sort=False):
        # #for spa,sdf in sig_df.groupby(["spa_version","nsa_mask_labels","spa_scale","tr_uuid","block_num","conv_ksize"]):
        # # for spa,sdf in sig_df.groupby(["spa_version","topk","spa_scale","asm"]):
        # # for spa,sdf in sig_df.groupby(["spa_version","topk","spa_scale"]):
        #     # print("spa: ",spa)
        #         print("SPA: ",spa,"%2.2f,%0.3f" %
        #               (sdf['psnrs'].mean(),sdf['ssims'].mean()))
            # print(sdf[['iname','psnrs','ssims']])

    df = results[results['dname']=='b100']
    df[['iname','sigma','spa_version','psnrs','ssims']].sort_values(["iname","sigma"]).to_csv("bsd100.csv",index=False)

    # for sigma,sig_df in results.groupby("sigma"):
    #     print("sigma: ",sigma)
    #     for spa,sdf in sig_df.groupby(["spa_version","topk","spa2_nsamples"]):
    #         # print(sdf[['dname','name','topk','psnrs','ssims']])
    #         print("SPA: ",spa,"%2.2f,%0.3f" %
    #               (sdf['psnrs'].mean(),sdf['ssims'].mean()))
    # #         # for dname,ddf in sdf.groupby("dname"):
    # #         #     # if dname != "set5": continue
    # #         #     print("[%s]: %2.2f,%0.3f" % (dname,ddf['psnrs'].mean(),ddf['ssims'].mean()))

    # for sigma,sig_df in results.groupby("sigma"):
    #     print("sigma: ",sigma)
    #     for f0,f0_df in sig_df.groupby(["spa2_kweight","spa2_normz"]):
    #         # print("f0: ",f0)
    #         print("f0: ",f0," %2.2f,%0.3f"%(f0_df['psnrs'].mean(),f0_df['ssims'].mean()))
    #         # for dname,ddf in f0_df.groupby("dname"):
    #             # print("[%s]: %2.2f,%0.3f" % (dname,ddf['psnrs'].mean(),ddf['ssims'].mean()))

if __name__ == "__main__":
    main()
