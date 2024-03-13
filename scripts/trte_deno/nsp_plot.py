import torch as th
import numpy as np
import pandas as pd
import cache_io
from easydict import EasyDict as edict
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt

def load_results(records_reload=False):

    # -- info --
    version = "v1"
    records_fn = ".cache_io_pkl/trte_deno/nsp_plot.pkl"
    cache_name = ".cache_io/trte_deno/test"

    # -- fast load --
    if not records_reload:
        try:
            cache = cache_io.ExpCache(cache_name,version)
            records = cache.to_records_fast(records_fn,False)
            return records
        except:
            pass

    # -- read slowly --
    read_testing = False
    refresh = records_reload and not(read_testing)
    train_fn_list = [
        "exps/trte_deno/train_nsp.cfg",
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
                                records_fn=".cache_io_pkl/trte_deno/nsp_plot.pkl",
                                records_reload=records_reload and not(read_testing),
                                use_wandb=False,proj_name="superpixels_deno_test")
    results = results.rename(columns={"affinity_softmax":"asm"})
    results = results.fillna(value=-1)
    results = results.drop_duplicates()

    return results

def format_data(df):
    # -- preformat --
    df = df.rename(columns={"affinity_softmax":"asm",})
    df = df.fillna(value=-1)
    df = df.drop_duplicates()
    attn_rn = {"nat":"NA","ssna":"SNA","sna":"H-SNA"}
    gsp_rn = {"none":"","default":"","modulated":"-SLIC","ssn":"-Deep"}

    # -- new df --
    fields = ["sigma","spa_version","stoken_size","gen_sp_type"]
    info = edict({"nsp":[],"psnrs":[],"ssims":[],"sigma":[],
                  "attn":[],"gsp":[],'stride':[]})
    for gfields,group in df.groupby(fields):
        sigma,attn,stride,gsp = gfields
        psnrs = group['psnrs'].mean().item()
        ssims = group['ssims'].mean().item()
        nsp = (512//stride)*(512//stride)//100
        info.nsp.append(nsp)
        info.psnrs.append(psnrs)
        info.ssims.append(ssims)
        info.sigma.append(sigma)
        info.stride.append(stride)
        # print(attn,gsp)
        # if attn == "nat":
        #     fixed_s = "Fixed $\lambda_{at}$"
        #     lrn_s = "Learn $\lambda_{at}$"
        #     suffix = lrn_s if lat else fixed_s
        #     info.attn.append(attn_rn[attn]+suffix)
        # else:
        info.attn.append(attn_rn[attn]+gsp_rn[gsp])
        info.gsp.append(gsp)
    info = pd.DataFrame(info)
    info = info.sort_values(["attn","sigma","nsp"])
    return info


def main():

    # -- get data --
    df = load_results(False)
    info = format_data(df)
    print(info)
    # info = info[info['attn'] != 'H-SNA']
    # info = info[info['gsp'] != 'modulated']

    # -- init --
    # ginfo = {'wspace':0.2, 'hspace':0.17,
    #          "top":0.90,"bottom":0.17,"left":0.14,"right":0.99}
    # fig,axes = plt.subplots(1,1,figsize=(4,2.75),gridspec_kw=ginfo)
    # axes = [axes]
    ginfo = {'wspace':0.22, 'hspace':0.19,
             "top":0.90,"bottom":0.18,"left":0.10,"right":0.99}
    fig,axes = plt.subplots(1,3,figsize=(8,2.75),gridspec_kw=ginfo)


    sigmas = [10,20,30]
    # sigmas = [10,20,30]
    # sigmas = [20]
    for i,sigma in enumerate(sigmas):
        ax = axes[i]
        info_i = info[info['sigma'] == sigma]
        sns.pointplot(data=info_i, x='nsp', y='psnrs', hue='attn',
                      # hue_order=["SNA","NA"],
                      # hue_order=["SNA","NA","H-SNA"],
                      # hue_order=["SNA-SLIC","SNA-Deep","NA"],
                      hue_order=["SNA-SLIC","SNA-Deep"],
                      ax=ax,# palette=mpl.colormaps["spring"],
                      # palette=sns.color_palette("spring", 6, as_cmap=True),
                      # hue_norm=mpl.colors.LogNorm(vmin=10.,vmax=30.),
                      markers='x', linewidth=2.)
        if sigma == 10:
            ax.get_legend().set_title("Method")
        else:
            ax.get_legend().remove()
        ax.set_title(r"$\sigma = "+str(sigma)+r"$",fontsize=12)
        if sigma == 10:
            ax.set_ylabel("PSNR (dB)",fontsize=12)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Num. of Superpixels ($10^2$)",fontsize=12)


        ymin = info_i['psnrs'].min().item()
        ymax = info_i['psnrs'].max().item()
        yticks = np.linspace(ymin,ymax,4)
        print(yticks)
        if i == 0:
            ylabels = ["%2.2f" % (y.round(decimals=2)) for y in yticks]
        else:
            ylabels = ["%2.1f" % (y.round(decimals=2)) for y in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

        xlabels = ax.get_xticklabels()
        ax.set_xticks(ax.get_xticks())
        xlabels = ax.get_xticklabels()
        # def fmt(txt): return r"$"+txt+r"^2$"
        def fmt(txt): return txt
        ax.set_xticklabels([fmt(x.get_text()) for x in xlabels])


    plt.savefig("output/figures/nsp_plot.png",dpi=500,transparent=True)

if __name__ == "__main__":
    main()

# import torch as th
# import numpy as np
# import pandas as pd
# import cache_io
# from easydict import EasyDict as edict

# import matplotlib.pyplot as plt

# def load_results(records_reload=False):
#     read_testing = False
#     refresh = False and not(read_testing)
#     tr_fn = "exps/trte_deno/train_nsp.cfg"
#     te_fn = "exps/trte_deno/test_shell.cfg"
#     tr_exp = cache_io.fill_test_shell(tr_fn,te_fn)
#     read_test = cache_io.read_test_config.run
#     _exps = read_test(tr_exp,".cache_io_exps/trte_deno/test",
#                       reset=refresh,skip_dne=refresh)
#     print(len(_exps))
#     exps,uuids = cache_io.get_uuids(_exps,".cache_io/trte_deno/test_v0",
#                                     read=not(refresh),no_config_check=False)
#     print(len(exps))
#     # -- run exps --
#     def dummy_fxn(*args): raise NotImplementedError("")
#     results = cache_io.run_exps(exps,dummy_fxn,uuids=uuids,preset_uuids=True,
#                                 name=".cache_io/trte_deno/test",
#                                 version="v1",skip_loop=True,
#                                 clear=False,enable_dispatch="slurm",
#                                 records_fn=".cache_io_pkl/trte_deno/nsp_plot.pkl",
#                                 records_reload=records_reload,# and not(read_testing),
#                                 use_wandb=False,proj_name="superpixels_deno_test")
#     results = results.rename(columns={"affinity_softmax":"asm"})
#     results = results.fillna(value=-1)
#     results = results.drop_duplicates()

#     return results

# def format_data(df):
#     fields = ["sigma","stoken_size"]
#     info = edict({"stride":[],"psnrs":[],"ssims":[]})
#     for gfields,group in df.groupby(fields):
#         stride = gfields[1]
#         psnrs = group['psnrs'].mean().item()
#         ssims = group['ssims'].mean().item()
#         info.stride.append(stride)
#         info.psnrs.append(psnrs)
#         info.ssims.append(ssims)
#     info = pd.DataFrame(info)
#     return info

# def main():

#     # -- get data --
#     df = load_results()
#     info = format_results(df)
#     print(info)

#     # -- unpack --
#     psnrs = info['psnrs'].to_numpy()
#     ssims = info['ssims'].to_numpy()
#     stride = info['stride'].to_numpy()



#     # -- init --
#     ginfo = {'wspace':0.05, 'hspace':0.17,
#              "top":0.93,"bottom":0.10,"left":0.14,"right":0.96}
#     fig,ax = plt.subplots(1,1,figsize=(4,2),gridspec_kw=ginfo)
#     ax.plot(stride,psnrs,'-x')
#     # ax.plot(stride,ssims,'-x')


# if __name__ == "__main__":
#     main()
