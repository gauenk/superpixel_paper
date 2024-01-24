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

    def forward(self,x,labels=None):
        # -- unpack superpixel info --
        if labels is None:
            sims, num_spixels = self.sp_sims(x)
            sims = rearrange(sims,'b s (h w) -> b h w',h=H)
        return self.neigh_sp_attn(x,labels)

# -- loading --
def load_ssna(nsp_version,dim,heads,qk_dim,gen_sp,**kwargs):
    cfg = edict(kwargs)
    ssna_cls = SoftSuperpixelNeighborhoodAttention
    neigh_sp_attn = ssn_cls(dim, heads,cfg.kernel_size,
                            qk_scale=cfg.spa_scale,
                            vweight=cfg.spa_vweight,
                            oweight=cfg.spa_oweight,
                            attn_normz=cfg.spa_attn_normz,
                            mask_labels=cfg.mask_labels)
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
