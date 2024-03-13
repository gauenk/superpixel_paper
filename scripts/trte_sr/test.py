
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
    read_testing = False

    # -- get/run experiments --
    refresh = True
    def clear_fxn(num,cfg): return False
    read_test = cache_io.read_test_config.run
    # exps = read_test("exps/trte_sr/test.cfg",
    #                  ".cache_io_exps/trte_sr/test",reset=refresh,skip_dne=refresh)
    # exps,uuids = cache_io.get_uuids(exps,".cache_io/trte_sr/test",
    #                                 read=not(refresh),no_config_check=False)
    # exps = list(reversed(exps))
    # uuids = list(reversed(uuids))
    # print("Num Exps: ",len(exps))
    # print(uuids)
    # # for e in exps:
    # #     print(e.upscale)

    # # -- get/run experiments --
    # refresh = False and not(read_testing)
    # def clear_fxn(num,cfg): return False
    # read_test = cache_io.read_test_config.run
    # # fn = "exps/trte_deno/test_early.cfg"
    # # fn = "exps/trte_deno/test_snet.cfg"
    # # fn = "exps/trte_deno/test.cfg"

    # -=-=-=-=-=-=-=-=-=-=-=-
    #
    # --  operator mode 1  --
    #
    # -=-=-=-=-=-=-=-=-=-=-=-

    train_fn_list = [
        # "exps/trte_sr/train_table.cfg",
        # "exps/trte_sr/train_ksize.cfg",
        # "exps/trte_sr/train_nsp.cfg",
        # "exps/trte_sr/train_snfts.cfg",
        # "exps/trte_sr/train_att_temp.cfg",
        "exps/trte_sr/train_att_temp_lrn.cfg",
        # "exps/trte_sr/train_bare.cfg",
    ]
    te_fn = "exps/trte_sr/test_shell.cfg"
    exps,uuids = [],[]
    for tr_fn in train_fn_list:
        tr_exp = cache_io.fill_test_shell(tr_fn,te_fn)
        _exps = read_test(tr_exp,".cache_io_exps/trte_sr/test",
                         reset=refresh,skip_dne=refresh)
        _exps,_uuids = cache_io.get_uuids(_exps,".cache_io/trte_sr/test_v0",
                                          read=not(refresh),no_config_check=False)
        exps += _exps
        uuids += _uuids

    print("Num Exps: ",len(exps))
    print(uuids)

    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_sr/test",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_sr/test.pkl",
                                records_reload=True and not(read_testing),
                                use_wandb=False,proj_name="superpixels_sr_test")

    # -- viz --
    results = results.rename(columns={"affinity_softmax":"asm"})
    results = results.fillna(value=-1)
    print(results[['spa_version','gen_sp_type']])
    print(results['spa_version'].unique())
    print(results.columns)
    results = results.drop_duplicates()
    for sigma,sig_df in results.groupby(["upscale","learn_attn_scale"]):
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
            ruuid = spa_df['pretrained_path'].unique().item()[:8]
            uuid = spa_df['uuid'].unique().item()[:8]
            print(ruuid)
            print("SPA: ",spa_version,"%2.2f,%0.3f" %
                  (spa_df['psnrs'].mean(),spa_df['ssims'].mean()))

    # # -- run exps --
    # results = cache_io.run_exps(exps,test.run,uuids=uuids,preset_uuids=True,
    #                             name=".cache_io/trte_sr/test",
    #                             version="v1",skip_loop=False,clear_fxn=clear_fxn,
    #                             clear=False,enable_dispatch="slurm",
    #                             records_fn=".cache_io_pkl/trte_sr/test.pkl",
    #                             records_reload=True,use_wandb=False,
    #                             proj_name="superpixels_sr_test")

    # results = results.rename(columns={"affinity_softmax":"asm"})
    # results = results.fillna(value=-1)
    # print(results[['spa_version','gen_sp_type']])
    # print(results['spa_version'].unique())
    # print(results['upscale'].unique())
    # for sigma,sig_df in results.groupby("upscale"):
    #     print("-"*20)
    #     print("upscale: ",sigma)
    #     print("-"*20)
    #     # for dname,ddf in sig_df.groupby("dname"):
    #     #     print("dname: ",dname)
    #     #     # print(ddf.columns)
    #     for spa_version,spa_df in sig_df.groupby(["spa_version","gen_sp_type","nsa_mask_labels","share_gen_sp","conv_ksize","use_ffn"]):
    #         # print(spa_version)
    #         # print(spa_df['psnrs'])
    #         # exit()
    #         print("SPA: ",spa_version,"%2.2f,%0.3f" %
    #               (spa_df['psnrs'].mean(),spa_df['ssims'].mean()))

    # # # print(results.columns)
    # # for spa,sdf in results.groupby("spa_version"):
    # #     print("SPA: ",spa)
    # #     # print(sdf[['dname','name','topk','psnrs','ssims']])
    # #     for dname,ddf in sdf.groupby("dname"):
    # #         # if dname != "set5": continue
    # #         print("[%s]: %2.2f,%0.3f" % (dname,ddf['psnrs'].mean(),ddf['ssims'].mean()))
    # #         # print(ddf[['name','topk','psnrs','ssims']])


if __name__ == "__main__":
    main()
