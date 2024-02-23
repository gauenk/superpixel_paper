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
from . import NeighborhoodSuperpixelAttention,SnaGmmSampling
# SnaGmmSampling,SnaSampling

class SNA(nn.Module):

    def __init__(self,sp_sims,neigh_sp_attn):
        super().__init__()
        self.sp_sims = sp_sims
        self.neigh_sp_attn = neigh_sp_attn

    def forward(self,x,labels=None):
        # -- unpack superpixel info --
        if labels is None:
            sims, num_spixels, _ = self.sp_sims(x)
            _,labels = torch.topk(sims.detach(),1,dim=-2)
            H = x.shape[-2]
            labels = rearrange(labels,'b 1 (h w) -> b h w',h=H)
        return self.neigh_sp_attn(x,labels)

# -- loading --
def load_sna(sna_version,dim,heads,qk_dim,gen_sp,**kwargs):
    cfg = edict(kwargs)

    qk_layer = kwargs['qk_layer'] if 'qk_layer' in kwargs else None
    v_layer = kwargs['v_layer'] if 'v_layer' in kwargs else None
    proj_layer = kwargs['proj_layer'] if 'proj_layer' in kwargs else None

    neigh_sp_attn = NeighborhoodSuperpixelAttention(dim, heads,# qk_dim,
                                                    cfg.kernel_size,
                                                    qk_scale=cfg.spa_scale,
                                                    mask_labels=cfg.mask_labels,
                                                    use_weights=cfg.use_weights,
                                                    use_proj=cfg.use_proj,
                                                    qk_layer=qk_layer,v_layer=v_layer,
                                                    proj_layer=proj_layer,
                                                    learn_attn_scale=cfg.learn_attn_scale)
              # normz_nsamples=cfg.spa_attn_normz_nsamples,
              # scatter_normz=cfg.spa_scatter_normz,
              # dist_type=cfg.dist_type)
    sna = SNA(gen_sp,neigh_sp_attn)
    if cfg.spa_full_sampling:
        # if cfg.sna_sim_method == "slic":
        if cfg.spa_sim_method == "slic":
            sna = SnaGmmSampling(sna,gen_sp,nsamples=cfg.spa_attn_nsamples)
        else:
            raise NotImplementedError("")
            # sna = SnaSampling(sna,gen_sp,nsamples=cfg.sna_attn_nsamples,topk=cfg.topk)
        # else:
        #     sna = SnaJointSampling(sna,gen_sp,nsamples=nsamples,topk=topk)
    return sna


