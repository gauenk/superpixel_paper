

# -- basic --
import math
import numpy as np
import torch as th
from einops import rearrange
from torchvision.utils import save_image,make_grid
from typing import Any, Callable
from easydict import EasyDict as edict
from dev_basics.utils.misc import ensure_chnls
from dev_basics.utils.metrics import compute_psnrs,compute_ssims,compute_strred

# -- plotting --
import pylab
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks

# -- better res --
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# -- forward processing --
from dev_basics import net_chunks
from dev_basics.utils import vid_io
from dev_basics.utils.misc import set_seed

# -- read in models --
import cache_io
from superpixel_paper.deno_trte import sr_utils as utils
from superpixel_paper.spa_config import config_via_spa
from superpixel_paper.deno_trte.train import extract_defaults as extract_deno_defaults
from superpixel_paper.deno_trte.train import seed_everything

# -- data --
import data_hub


class AttentionHook():

    def __init__(self,net):
        self.net = net
        # print([name for name,_ in self.net.named_modules()])

        # -- known buffer names --
        self.bufs = ["q_shell","k_shell","v_shell","attn_shell",
                     "imgSp_shell_attn","imgSp_shell_agg","spa_shell",
                     "skipped_x_shell","pre_layernorm","post_layernorm",
                     "pre_conv","post_conv"]
        for buf in self.bufs:
            setattr(self,buf,[])

        # -- add hook --
        for name,layer in self.net.named_modules():
            for buf in self.bufs:
                if buf in name:
                    layer.register_forward_hook(self.save_outputs_hook(buf,name))

    def save_outputs_hook(self, buffer_name: str, layer_id: str) -> Callable:
        buff = getattr(self,buffer_name)
        def fn(_, __, output):
            buff.append(output)
        return fn

    # def unpack(self):
    #     "q_shell","k_shell","v_shell",
    #     "imgSP_shell_attn","imgSp_shell_agg"
    #     for buf in self.bufs:
    #     print(hooks[keys[0]].q_shell)


def load_video(cfg):
    cfg.dname = "set8"
    cfg.dset = "val"
    cfg.vid_name = "sunflower"
    cfg.nframes = 1
    cfg.batch_size_tr = 1
    device = "cuda:0"
    cfg.nsamples_val = 0
    data,loaders = data_hub.sets.load(cfg)
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,0,cfg.nframes)
    vid = data[cfg.dset][indices[0]]['clean'][None,:].to(device)#/255.
    noisy = data[cfg.dset][indices[0]]['noisy'][None,:].to(device)#/255.
    return noisy[:,0]

def apply_hooks(net):
    hook = AttentionHook(net)
    return hook

def sp_to_mask(imgSp):
    # -- get masks --
    uniqs = imgSp.unique()
    masks = []
    for u in uniqs:
        masks.append(imgSp==u)
    masks = th.cat(masks)
    return masks


def draw_seg(img,imgSp):
    ncols = 1500
    # cm = pylab.get_cmap('gist_rainbow')
    cm = pylab.get_cmap('prism')
    def index2color(i):
        nmax = 200
        cm_i = ((1.*(i%nmax))/nmax) % 1
        color = [int(255.*i.item()) for i in cm(cm_i)]
        color = [color[0],color[1],color[2]]
        return tuple(color)

    # -- masks --
    masks = sp_to_mask(imgSp)
    img_ui = th.clamp(255*img,0,255).type(th.uint8)[0].cpu()
    color_map = [index2color(i%ncols) for i in range(ncols)]
    seg_result = draw_segmentation_masks(
        img_ui*0, masks.cpu(), alpha=0.3,
        # colors="blue",
        colors=color_map,
    )
    return seg_result.to(img.device)

# def nicer_image(vid):
#     B = vid.shape[0]
#     H,W = vid.shape[-2:]
#     cH = 6*H
#     cW = 6*W
#     ndim = vid.ndim
#     if ndim == 5:
#         vid = rearrange(vid,'b t ... -> (b t) ...')
#     vid = TF.resize(vid,(cH,cW),InterpolationMode.NEAREST)
#     if ndim == 5:
#         vid = rearrange(vid,'(b t) ... -> b t ...',b=B)
#     return vid

def main():


    # -=-=-=-=-=-=-=-=-=-
    #
    # --     Setup     --
    #
    # -=-=-=-=-=-=-=-=-=-

    # -- seed --
    seed = 123
    seed_everything(seed)

    # -- read config --
    # fn_a = "exps/trte_deno/viz_attn_layer.cfg"
    fn_a = "exps/trte_deno/train.cfg"
    fn_b = ".cache_io_exps/trte_deno/viz_attn_layer/"
    refresh = True
    read_test = cache_io.read_test_config.run
    # exps = read_test(fn_a,fn_b,reset=refresh,skip_dne=refresh)
    # exps,_uuids = cache_io.get_uuids(exps,".cache_io_exps/viz_attn_layer/",
    #                                 read=not(refresh),no_config_check=False)
    exps,_uuids = cache_io.train_stages.run(fn_a,fn_b,update=True)

    # -- load models --
    device = "cuda:0"
    cfgs,uuids = [],[]
    models,hooks = {},{}
    for uuid,exp in zip(_uuids,exps):
        uuid_s = str(uuid)[:4]
        cfg = extract_deno_defaults(exp)
        # cfg.M = 0.001
        # cfg.M = 0.1
        # cfg.M = 0.02
        # cfg.M = 0.5
        # cfg.M = 10.
        # cfg.M = 0.002
        config_via_spa(cfg)
        uuids.append(uuid_s)
        cfgs.append(cfg)
        # print(uuid_s,cfg.spa_version,cfg.spa_scale,cfg.nsa_mask_labels)
        print(uuid_s,cfg.nsa_mask_labels,cfg.use_skip)
        import_str = 'superpixel_paper.models.{}'.format(cfg.model)
        model = utils.import_module(import_str).create_model(cfg).to(device)
        hook = apply_hooks(model)
        models[uuid_s] = model
        hooks[uuid_s] = hook
        # break

    # -- load hooks with values --
    keys = uuids
    vid = load_video(cfg)
    # vid = vid[...,190:190+156,400:400+156].contiguous()
    vid = vid[...,:,128:-128].contiguous()
    print(vid.shape)
    for key in keys: models[key](vid)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    # --    Explore; Set Sail!    --
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- hooks --
    vid_m = vid.mean(1,keepdim=True)/255.
    # print(vid.shape,vid_m.shape)
    # print(hooks[keys[0]].spa_shell[0].shape)
    dim_sel = 1
    print([hooks[k].spa_shell[0][0,[dim_sel],64:68,64:68] for k in keys])
    def est_var(lname):
        est_vars = {}
        for k in keys:
            est_var = 0
            for i in range(9):
                est_var += getattr(hooks[k],lname)[0][0,[dim_sel]].var().item()
            est_vars[k] = est_var/9
        print(est_vars)
    print("-"*30)
    print("-"*30)
    print((vid/255.).var().item())
    est_var('pre_conv')
    est_var('post_conv')
    print("-"*10)
    est_var('pre_layernorm')
    est_var('post_layernorm')
    print("-"*10)
    est_var('spa_shell')
    est_var('skipped_x_shell')
    print("-"*30)
    print("-"*30)
    # print([hooks[k].spa_shell[0][0,[dim_sel]].var() for k in keys])
    spa_out = th.cat([vid_m]+[hooks[k].spa_shell[0][:,[dim_sel]] for k in keys])
    eps = 1e-6
    spa_out = th.stack([x.abs()/(x.abs().max()+eps) for x in spa_out])
    print([x.max() for x in spa_out])
    nrow = len(spa_out)
    print(spa_out.shape)
    grid = make_grid(spa_out,nrow=nrow,pad_value=0.)[None,:]
    save_image(grid/grid.max(),"spa.png")

    # -- explore sp --
    imgSp = th.stack([draw_seg(vid,hooks[k].imgSp_shell_attn[0]) for k in keys])
    print(imgSp.shape)
    nrow = len(imgSp)
    grid = make_grid(imgSp,nrow=nrow,pad_value=1.)[None,:]
    print(grid.shape)
    save_image(grid/grid.max(),"grid_sp.png")
    # exit()

    # -- explore attn --
    attns = th.stack([hooks[k].attn_shell[0] for k in keys])
    attns = [th.softmax(30*attns[i],-1) for i in range(len(attns))]
    eps = 1e-3
    # print(attns.max())
    # print([attns[i].max().item() for i in range(len(attns))])
    # for i in range(len(attns)):
    #     max_i = attns[i].max().item()
    #     min_i = attns[i].min().item()
    #     tmp = attns[i]/(attns[i].abs().max().item()+1e-5)
    #     print(i,max_i,min_i,tmp.max(),tmp.min())
    # exit()
    attns = th.stack([attns[i] for i in range(len(attns))])
    print(attns.max())
    K = int(math.sqrt(attns.shape[-1]))
    attns = rearrange(attns,'g 1 1 h w (k0 k1) -> g h w k0 k1',k0=K)
    print(attns.shape)

    # -- save viz --
    points = [[250,64],[250,96],[228,128]]
    for point in points:
        x,y = point
        nrow = attns.shape[0]
        print(attns[:,x,y,:3,:3])
        grid = make_grid(attns[:,x,y],nrow=nrow,pad_value=1.)[:,None]
        print(grid.shape)
        save_image(grid/grid.max(),"grid_%d_%d.png" % (x,y))
    # grid = make_grid(attns[:,228,128],nrow=nrow,pad_value=1.)[:,None]
    # print(grid.shape)
    # save_image(grid/grid.max(),"grid_228_128.png")

    print(attns.shape)
    grid = make_grid(attns[:,:,:,3,3],nrow=nrow,pad_value=1.)[:,None]
    print(grid.shape)
    print(grid.max())
    save_image(grid/grid.max(),"grid_hw_0.png")

    print("vid.shape: ",vid.shape)

    for point in points:
        x,y = point
        vid[:,:,x-2:x+2,y-2:y+2] = 1
    save_image(vid,"vid.png")



if __name__ == "__main__":
    main()
