# -- import torch --
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# -- basic --
import math
from einops import rearrange
from easydict import EasyDict as edict

# -- local --
from .spa_utils import unpack_sp,mask_from_labels,weighted_scatter_mean
from .spa_utils import normalize_attention,estimate_normz,scatter_mean
from .guts import LayerNorm2d


#
#
# -- Superpixel Attention --
#
#

class SPASampling(nn.Module):

    def __init__(self, dim, num_heads, qk_dim, sim_sp, topk=32,
                 qkv_bias=False, qk_scale=None, attn_drop=0.,
                 normz=False,kweight=True,out_weight=True,nsamples=30,
                 normz_version="map"):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.num_heads = num_heads
        self.nsamples = nsamples
        self.topk = topk
        self.normz_version = normz_version

        self.q = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, 1, bias=qkv_bias)

        self.norm = LayerNorm2d(dim)
        self.normz = normz
        self.kweight = kweight
        self.out_weight = out_weight

        self.attn_drop = nn.Dropout(attn_drop)
        head_dim = self.qk_dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

    def forward(self, x, joint_probs):


        # -- compute the q,k,v --
        B, C, H, W = x.shape
        x = self.norm(x)
        q, k, v = self.q(x), self.k(x), self.v(x)

        # -- unpack superpixel info --
        if (affinity_matrix is None) and (labels is None):
            affinity_matrix, num_spixels = self.sim_sp(x)
        if labels is None:
            sims, indices, labels, mask = unpack_sp(affinity_matrix,self.topk)
        joint_probs = get_joint_est(samples)
        mask = None

        # -- create q,k,v --
        sample_it = lambda qkv,dim: torch.gather(
            qkv.reshape(B, 1, -1, H*W).expand(-1, num_spixels, -1, -1),
            -1, indices.unsqueeze(2).expand(-1, -1, dim, -1)) # B, K, C, topk
        shape_str = 'b k (h c) t -> b k h t c'
        reshape_it = lambda qkv: rearrange(qkv, shape_str, h=self.num_heads)
        Q_sp = reshape_it(sample_it(q,self.qk_dim))
        K_sp = reshape_it(sample_it(k,self.qk_dim))
        V_sp = reshape_it(sample_it(v,self.dim))

        # -- compute attention --
        attn = (Q_sp @ K_sp.transpose(-2,-1)) * self.scale # b k h topk topk
        attn = joint_probs[:,:,None] * attn
        attn = normalize_attention(self.normz_version,attn,sims,mask)
        attn = self.attn_drop(attn)
        out = attn @ V_sp # b k h topk c

        #
        # -- Scattering back to img --
        #

        out = rearrange(out, 'b k h t c -> b (h c) (k t)')
        weight = rearrange(weight,'b k h t c -> b (h c) (k t)')
        out = weighted_scatter_mean(v.reshape(B, self.dim, H*W), weight, None, -1,
                                    indices.reshape(B, 1, -1).expand(-1, self.dim, -1),
                                    out,normz=self.normz)
        out = out.reshape(B, C, H, W)

        return out


#
#
# -- Sampling to Estimate Joint --
#
#

class SpaJointSampling(nn.Module):
    def __init__(self, attn, sim_sp, attn_nsamples=10, sp_nsamples=10, topk=32):
        super().__init__()
        self.attn = attn
        self.attn_nsamples = attn_nsamples
        self.sp_nsamples = sp_nsamples
        self.topk = topk
        self.sim_sp = sim_sp

    def forward(self, x):

        # -- prepare --
        samples = self.sim_sp(x,self.sp_nsamples)
        out = self.attn(x,samples)/self.attn_nsamples

        # -- compute average --
        for _ in range(self.attn_nsamples-1):

            # -- samples --
            samples = self.sim_sp(x,self.sp_nsamples)
            out += self.attn(x,samples)/self.attn_nsamples
        return out

