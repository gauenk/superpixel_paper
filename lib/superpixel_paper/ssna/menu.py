# -- import torch --
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# -- basic --
import math
from einops import rearrange
from easydict import EasyDict as edict

# -- modules --
from .ssna import SoftSuperpixelNeighborhoodAttention
# NspGmmSampling,NspSampling

class SSNA(nn.Module):

    def __init__(self,sp_sims,neigh_sp_attn):
        super().__init__()
        self.sp_sims = sp_sims
        self.neigh_sp_attn = neigh_sp_attn

    def forward(self,x,sims=None,state=None):
        # -- unpack superpixel info --
        if sims is None:
            sims, num_spixels, state = self.sp_sims(x,state=state)
        return self.neigh_sp_attn(x,sims),state

# -- loading --
def load_ssna(nsp_version,dim,heads,qk_dim,gen_sp,**kwargs):
    cfg = edict(kwargs)
    qk_layer = kwargs['qk_layer'] if 'qk_layer' in kwargs else None
    v_layer = kwargs['v_layer'] if 'v_layer' in kwargs else None
    proj_layer = kwargs['proj_layer'] if 'proj_layer' in kwargs else None

    ssna_cls = SoftSuperpixelNeighborhoodAttention
    neigh_sp_attn = ssna_cls(dim, heads,cfg.kernel_size,
                             qk_scale=cfg.spa_scale,
                             mask_labels=cfg.mask_labels,
                             use_weights=cfg.use_weights,
                             use_proj=cfg.use_proj,
                             qk_layer=qk_layer,v_layer=v_layer,
                             proj_layer=proj_layer,
                             learn_attn_scale=cfg.learn_attn_scale,
                             detach_sims=cfg.detach_sims,
                             detach_learn_attn=cfg.detach_learn_attn,
                             attn_rw_version=cfg.attn_rw_version)
    ssna = SSNA(gen_sp,neigh_sp_attn)
    # if cfg.spa_full_sampling:
    #     # if cfg.nsp_sim_method == "slic":
    #     if cfg.spa_sim_method == "slic":
    #         nsp = NspGmmSampling(nsp,gen_sp,nsamples=cfg.spa_attn_nsamples)
    #     else:
    #         raise NotImplementedError("")
    #         # nsp = NspSampling(nsp,gen_sp,nsamples=cfg.nsp_attn_nsamples,topk=cfg.topk)
    #     # else:
    #     #     nsp = NspJointSampling(nsp,gen_sp,nsamples=nsamples,topk=topk)
    return ssna
