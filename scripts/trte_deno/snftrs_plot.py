import torch as th
import numpy as np
import pandas as pd
import cache_io
from easydict import EasyDict as edict
import seaborn as sns
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

def load_results(records_reload=False):

    # -- info --
    version = "v1"
    fmt_fn = ".cache_io_pkl/trte_deno/sftrs_plot_fmt.pkl"
    records_fn = ".cache_io_pkl/trte_deno/sftrs_plot.pkl"
    cache_name = ".cache_io/trte_deno/test"

    # -- fast load --
    if not records_reload:
        try:
            if Path(fmt_fn).exists():
                results = pd.read_pickle(fmt_fn)
                return results
            else:
                print("c")
                cache = cache_io.ExpCache(cache_name,version)
                results = cache.to_records_fast(records_fn,False)
                print("d")
                results = results.rename(columns={"affinity_softmax":"asm"})
                results = results.fillna(value=-1)
                results = results.drop_duplicates()
                print("e")
                fill_num_params(results)
                print("f")
                results.to_pickle(fmt_fn)
        except:
            pass

    # -- read slowly --
    # read_testing = True
    # read_testing = False
    # refresh = records_reload and not(read_testing)
    # if not(read_testing):
    #     def dummy_fxn(*args): raise NotImplementedError("")
    #     results = cache_io.run_exps([],dummy_fxn,name=cache_name,
    #                                 version=version,skip_loop=True,
    #                                 clear=False,enable_dispatch="slurm",
    #                                 records_fn=records_fn,
    #                                 records_reload=records_reload and not(read_testing),
    #                                 use_wandb=False,proj_name="superpixels_deno_test")

    read_testing = False
    refresh = records_reload and not(read_testing)
    train_fn_list = [
        "exps/trte_deno/train_snfts.cfg",
        "exps/trte_deno/train_snfts_nat.cfg"
    ]
    te_fn = "exps/trte_deno/test_shell.cfg"
    # te_fn = "exps/trte_deno/test_shell_bsd68.cfg"
    exps,uuids = [],[]
    read_test = cache_io.read_test_config.run
    for tr_fn in train_fn_list:
        tr_exp = cache_io.fill_test_shell(tr_fn,te_fn)
        _exps = read_test(tr_exp,".cache_io_exps/trte_deno/test",
                          reset=refresh,skip_dne=refresh)
        _exps,_uuids = cache_io.get_uuids(_exps,".cache_io/trte_deno/test_v0",
                                          read=not(refresh),no_config_check=False)
        # _exps,_uuids = cache_io.get_uuids(_exps,".cache_io/trte_deno/test",
        #                                   read=not(refresh),no_config_check=False)
        exps += _exps
        uuids += _uuids
    print("Num Exps: ",len(exps))
    # print(uuids)
    # te_fn = "exps/trte_deno/test_shell.cfg"
    # tr_exp = cache_io.fill_test_shell(tr_fn,te_fn)
    # read_test = cache_io.read_test_config.run
    # _exps = read_test(tr_exp,".cache_io_exps/trte_deno/test",
    #                   reset=refresh,skip_dne=refresh)
    # exps,uuids = cache_io.get_uuids(_exps,".cache_io/trte_deno/test_v0",
    #                                 read=not(refresh),no_config_check=False)

    # -- run exps --
    def dummy_fxn(*args): raise NotImplementedError("")
    results = cache_io.run_exps(exps,dummy_fxn,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_deno/test",
                                version="v1",skip_loop=True,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_deno/sftrs_plot.pkl",
                                records_reload=records_reload and not(read_testing),
                                use_wandb=False,proj_name="superpixels_deno_test")
    results = results.rename(columns={"affinity_softmax":"asm"})
    results = results.fillna(value=-1)
    results = results.drop_duplicates()
    fill_num_params(results)
    results.to_pickle(".cache_io_pkl/trte_deno/sftrs_plot_fmt.pkl")

    return results

# def format_data(df):
#     # -- preformat --
#     df = df.rename(columns={"affinity_softmax":"asm"})
#     df = df.fillna(value=-1)
#     df = df.drop_duplicates()

#     # -- new df --
#     fields = ["sigma","spa_version","nat_ksize"]
#     info = edict({"ksize":[],"psnrs":[],"ssims":[],"sigma":[],"attn":[]})
#     for gfields,group in df.groupby(fields):
#         sigma,attn,ksize = gfields
#         psnrs = group['psnrs'].mean().item()
#         ssims = group['ssims'].mean().item()
#         info.ksize.append(ksize)
#         info.psnrs.append(psnrs)
#         info.ssims.append(ssims)
#         info.sigma.append(sigma)
#         info.attn.append(attn)
#     info = pd.DataFrame(info)
#     info = info.sort_values(["attn","sigma","ksize",])
#     return info


def format_data(df):
    # -- preformat --
    df = df.rename(columns={"affinity_softmax":"asm","learn_attn_scale":"las"})
    df = df.fillna(value=-1)
    df = df.drop_duplicates()
    attn_rn = {"nat":"NA","ssna":"SNA","sna":"H-SNA"}
    gsp_rn = {"none":"","default":"","modulated":"-SLIC","ssn":"-Deep"}

    # -- new df --
    fields = ["sigma","spa_version","ssn_nftrs","gen_sp_type","las",'nparams']
    info = edict({"nparams":[],"psnrs":[],"ssims":[],"sigma":[],
                  "attn":[],"gsp":[]})
    for gfields,group in df.groupby(fields):
        sigma,attn,nftrs,gsp,las,nparams = gfields
        psnrs = group['psnrs'].mean().item()
        ssims = group['ssims'].mean().item()
        # info.nftrs.append(nftrs)
        info.psnrs.append(psnrs)
        info.ssims.append(ssims)
        info.sigma.append(sigma)
        info.nparams.append(nparams)
        # print(attn,gsp)
        # info.attn.append(attn_rn[attn]+gsp_rn[gsp])
        if attn == "nat":
            fixed_s = "-Fixed"
            lrn_s = "-Learn"
            suffix = lrn_s if las else fixed_s
            info.attn.append(attn_rn[attn]+suffix)
        else:
            info.attn.append(attn_rn[attn]+gsp_rn[gsp])
        info.gsp.append(gsp)
        # info.las.append(las)
    info = pd.DataFrame(info)
    info = info.sort_values(["attn","sigma","nparams"])
    return info


def fill_num_params(df):
    nparams = []
    for _,row in df.iterrows():
        row_d = edict(row.to_dict())
        nparams.append(get_num_params(row_d))
    df['nparams'] = nparams

def get_num_params(_cfg):
    # -- load model --
    from superpixel_paper.deno_trte.train import load_model,extract_defaults
    from superpixel_paper.spa_config import config_via_spa
    _cfg = extract_defaults(_cfg)
    config_via_spa(_cfg,False)
    model = load_model(_cfg)
    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    nparams = num_parameters / 10 ** 3
    return nparams

def main():

    # -- get data --
    df = load_results(False)
    # fill_num_params(df)
    df['nparams'] = df['nparams'].round().astype(int)
    print(df)
    info = format_data(df)
    # info = info[info['las'] != True]
    # info = info[info['attn'] != 'H-SNA']
    info = info[info['attn'] != 'NA-Fixed']
    # info = info[info['gsp'] != 'modulated']
    print(info)

    # -- init --
    # ginfo = {'wspace':0.2, 'hspace':0.17,
    #          "top":0.90,"bottom":0.17,"left":0.14,"right":0.99}
    # fig,axes = plt.subplots(1,1,figsize=(4,2.75),gridspec_kw=ginfo)
    # axes = [axes]
    ginfo = {'wspace':0.2, 'hspace':0.17,
             "top":0.90,"bottom":0.17,"left":0.08,"right":0.99}
    fig,axes = plt.subplots(1,3,figsize=(8,2.75),gridspec_kw=ginfo)


    sigmas = [10,20,30]
    # sigmas = [20]
    for i,sigma in enumerate(sigmas):
        ax = axes[i]
        info_i = info[info['sigma'] == sigma]
        # print(info_i)
        sns.pointplot(data=info_i, x='nparams', y='psnrs', hue='attn',
                      # hue_order=["SNA","NA"],
                      # hue_order=["SNA","NA","H-SNA"],
                      hue_order=["SNA-SLIC","SNA-Deep","NA-Learn"],
                      ax=ax,# palette=mpl.colormaps["spring"],
                      # palette=sns.color_palette("spring", 6, as_cmap=True),
                      # hue_norm=mpl.colors.LogNorm(vmin=10.,vmax=30.),
                      markers='x', linewidth=2.)
        if sigma == 10:
            sns.move_legend(ax, "center left")
            ax.get_legend().set_title("Method")
        else:
            ax.get_legend().remove()
        ax.set_title(r"$\sigma = "+str(sigma)+r"$",fontsize=12)
        if sigma == 10:
            ax.set_ylabel("PSNR (dB)",fontsize=12)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Aux Net Size ($10^3$ Params)",fontsize=12)


        ymin = info_i['psnrs'].min().item()
        ymax = info_i['psnrs'].max().item()
        yticks = np.linspace(ymin,ymax,5)
        ylabels = ["%2.1f" % y for y in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

        xlabels = ax.get_xticklabels()
        ax.set_xticks(ax.get_xticks())
        xlabels = ax.get_xticklabels()
        def fmt(txt):
            if int(txt) == 149: return "150"
            elif int(txt) == 67: return "70"
            elif int(txt) == 38: return "40"
            elif int(txt) == 5: return "5"
            else: return txt
        ax.set_xticklabels([fmt(x.get_text()) for x in xlabels])


    plt.savefig("output/figures/snftrs_plot.png",dpi=500,transparent=True)


if __name__ == "__main__":
    main()
