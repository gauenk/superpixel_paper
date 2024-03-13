
import torch as th
import pandas as pd
from easydict import EasyDict as edict

# -- caching --
import cache_io

# -- bench --
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt


def run_loss(model,name,loss_fxn):

    # -- timer/memer --
    timer = ExpTimer()
    memer = GpuMemer()

    # -- init data --
    size = 128
    noisy = th.zeros((1,3,size,size),device="cuda")
    seg = (th.rand((1,size,size),device="cuda")*10).long()

    # -- add hooks --
    from superpixel_paper.models.sp_hooks import SsnaSuperpixelHooks
    sphooks = None
    if name != "deno":
        sphooks = SsnaSuperpixelHooks(model)

    # -- bench fwd --
    print(name)
    # exit()
    with TimeIt(timer,"run"):
        with MemIt(memer,"run"):
            deno = model(noisy)
            if name in ["seg","pix"]:
                B = noisy.shape[0]
                HW = deno.shape[-2]*deno.shape[-1]
                sims = sphooks.spix[0]
                sims = sims.reshape(B,HW,-1).transpose(2,1)
                if name == "seg":
                    loss = loss_fxn(seg[:,None],sims)
                elif name == "pix":
                    loss = loss_fxn(noisy,sims)
            else:
                loss = loss_fxn(deno,noisy)
            loss.backward()

    # -- results --
    info = edict()
    info['time'] = timer['run']
    info['mem'] = memer['run']['alloc']
    info['name'] = name

    return info

def load_model():
    # -- load config --
    model_uuid = "ac0326d3-a2f9-4f00-8200-3eafa39c19c8"
    cache = cache_io.ExpCache(".cache_io/trte_deno/train","v1")
    _cfg = cache.get_config_from_uuid(model_uuid)
    assert (_cfg != -1),f"Invalid uuid. Check original training script. [{model_uuid}]"

    # -- load config --
    from superpixel_paper.deno_trte.train import load_model,extract_defaults
    from superpixel_paper.spa_config import config_via_spa
    _cfg = extract_defaults(_cfg)
    config_via_spa(_cfg)
    model = load_model(_cfg).cuda()
    return model

def main():


    # -- init losses --
    eps = 1e-3
    losses = edict()
    losses.warmup = lambda deno,clean: th.sqrt(th.mean((deno-clean)**2)+eps**2)
    losses.deno = lambda deno,clean: th.sqrt(th.mean((deno-clean)**2)+eps**2)
    from superpixel_paper.ssn_trte.label_loss import SuperpixelLoss
    losses.seg = SuperpixelLoss("cross")
    losses.pix = SuperpixelLoss("mse")

    # -- load model --
    model = load_model()
    model.train()

    # -- run --
    info = {"time":[],"mem":[],"name":[]}
    for name,loss in losses.items():
        info_n = run_loss(model,name,loss)
        for field in info:
            info[field].append(info_n[field])
        print(info_n)
    info = pd.DataFrame(info)
    print(info)


if __name__ == "__main__":
    main()
