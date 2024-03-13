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
    records_fn=".cache_io_pkl/trte_deno/neigh_size_plot.pkl"
    cache_name = ".cache_io/trte_deno/test"

    # -- fast load --
    if not records_reload:
        try:
            cache = cache_io.ExpCache(cache_name,version)
            records = cache.to_records_fast(records_fn,False)
            # records = cache.to_records_fast(records_fn,False)
            return records
        except:
            pass

    # -- read slowly --
    read_testing = False
    refresh = records_reload and not(read_testing)
    train_fn_list = [
        "exps/trte_deno/train_ksize.cfg",
        "exps/trte_deno/train_ksize_nat.cfg",
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
        exps += _exps
        uuids += _uuids

    # tr_fn = "exps/trte_deno/train_ksize.cfg"
    # te_fn = "exps/trte_deno/test_shell.cfg"
    # tr_exp = cache_io.fill_test_shell(tr_fn,te_fn)
    # read_test = cache_io.read_test_config.run
    # _exps = read_test(tr_exp,".cache_io_exps/trte_deno/test",
    #                   reset=refresh,skip_dne=refresh)
    # exps,uuids = cache_io.get_uuids(_exps,".cache_io/trte_deno/test_v0",
    #                                 read=not(refresh),no_config_check=False)

    # -- run exps --
    def dummy_fxn(*args): raise NotImplementedError("")
    results = cache_io.run_exps(exps,dummy_fxn,preset_uuids=True,
                                name=cache_name,version=version,skip_loop=True,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_deno/ksize_plot.pkl",
                                records_reload=records_reload,# and not(read_testing),
                                use_wandb=False,proj_name="superpixels_deno_test")

    return results

def format_data(df):
    # -- preformat --
    df = df.rename(columns={"affinity_softmax":"asm"})
    df = df.fillna(value=-1)
    df = df.drop_duplicates()
    attn_rn = {"nat":"NA-Fixed","ssna":"SNA-Deep","sna":"H-SNA"}
    # -- new df --
    fields = ["sigma","spa_version","nat_ksize","gen_sp_type"]
    info = edict({"ksize":[],"psnrs":[],"ssims":[],"sigma":[],
                  "attn":[],"gsp":[]})
    for gfields,group in df.groupby(fields):
        sigma,attn,ksize,gsp = gfields
        psnrs = group['psnrs'].mean().item()
        ssims = group['ssims'].mean().item()
        info.ksize.append(ksize)
        info.psnrs.append(psnrs)
        info.ssims.append(ssims)
        info.sigma.append(sigma)
        info.attn.append(attn_rn[attn])
        info.gsp.append(gsp)
    info = pd.DataFrame(info)
    info = info.sort_values(["attn","sigma","ksize"])
    return info

def main():

    # -- get data --
    df = load_results(False)
    info = format_data(df)
    print(info)
    # info = info[info['attn'] != 'H-SNA']
    info = info[info['gsp'] != 'modulated']

    # -- x-axis --
    psnrs = df['psnrs'].to_numpy()
    max_y = psnrs.max().item()
    min_y = psnrs.min().item()
    print(max_y,min_y)

    # -- unpack --
    # psnrs = info['psnrs'].to_numpy()
    # ssims = info['ssims'].to_numpy()
    # ksize = info['ksize'].to_numpy()
    # info = info[info['sigma'] == 20]

    # -- init --
    ginfo = {'wspace':0.2, 'hspace':0.17,
             "top":0.90,"bottom":0.17,"left":0.08,"right":0.99}
    fig,axes = plt.subplots(1,3,figsize=(8,2.75),gridspec_kw=ginfo)

    sigmas = [10,20,30]
    for i,sigma in enumerate(sigmas):
        ax = axes[i]
        info_i = info[info['sigma'] == sigma]
        sns.pointplot(data=info_i, x='ksize', y='psnrs', hue='attn',
                      # hue_order=["SNA","NA"],
                      hue_order=["SNA-Deep","NA-Fixed","H-SNA"],
                      ax=ax,# palette=mpl.colormaps["spring"],
                      # palette=sns.color_palette("spring", 6, as_cmap=True),
                      # hue_norm=mpl.colors.LogNorm(vmin=10.,vmax=30.),
                      markers='x', linewidth=2.)
        if sigma == 10:
            sns.move_legend(ax, "lower left")
            ax.get_legend().set_title("Method")
        else:
            ax.get_legend().remove()
        ax.set_title(r"$\sigma = "+str(sigma)+r"$",fontsize=12)
        if sigma == 10:
            ax.set_ylabel("PSNR (dB)",fontsize=12)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Neighborhood Size (pixels)",fontsize=12)


        ymin = info_i['psnrs'].min().item()
        ymax = info_i['psnrs'].max().item()
        yticks = np.linspace(ymin,ymax,5)
        ylabels = ["%2.1f" % y for y in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

        xlabels = ax.get_xticklabels()
        ax.set_xticks(ax.get_xticks())
        xlabels = ax.get_xticklabels()
        def fmt(txt): return r"$"+txt+r"^2$"
        ax.set_xticklabels([fmt(x.get_text()) for x in xlabels])


    # xticks = ax.get_xticks()
    # ax.set_xticklabels([x_str+"^2" for x_str in xticks])
    # ax.plot(ksize,psnrs,'-x')
    # # ax.plot(stride,ssims,'-x')

    plt.savefig("output/figures/neigh_size_plot.png",dpi=500,transparent=True)

if __name__ == "__main__":
    main()
