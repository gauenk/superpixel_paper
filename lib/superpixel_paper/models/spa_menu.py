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
from .spa_modules import SPA,SpaGmmSampling,SpaSampling

# -- loading --
def load_spa(spa_version,dim,heads,qk_dim,gen_sp,**kwargs):
    cfg = edict(kwargs)
    spa = SPA(dim, heads, qk_dim, gen_sp,
              topk=cfg.topk,qk_scale=cfg.spa_scale,
              vweight=cfg.spa_vweight,out_weight=cfg.spa_oweight,
              attn_normz=cfg.spa_attn_normz,
              normz_nsamples=cfg.spa_attn_normz_nsamples,
              scatter_normz=cfg.spa_scatter_normz,
              dist_type=cfg.dist_type)
    if cfg.spa_full_sampling:
        if cfg.spa_sim_method == "slic":
            spa = SpaGmmSampling(spa,gen_sp,nsamples=cfg.spa_attn_nsamples,topk=cfg.topk)
        else:
            spa = SpaSampling(spa,gen_sp,nsamples=cfg.spa_attn_nsamples,topk=cfg.topk)
        # else:
        #     spa = SpaJointSampling(spa,gen_sp,nsamples=nsamples,topk=topk)
    return spa


