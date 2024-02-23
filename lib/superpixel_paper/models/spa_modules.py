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

# -- superpixel --
from superpixel_paper.est_attn_normz import EffNormzFunction
from ..sr_models.pair_wise_distance import PairwiseDistFunction
from natten import NeighborhoodAttention2D


#
#
# -- Superpixel Attention --
#
#

class SPA(nn.Module):

    def __init__(self, dim, num_heads, qk_dim, est_marginal_sp,
                 topk=32, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 vweight=True,out_weight=True,
                 attn_normz="map",normz_nsamples=10,scatter_normz=False,
                 dist_type="l2"):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.num_heads = num_heads
        self.topk = topk
        self.attn_normz = attn_normz
        self.normz_nsamples = normz_nsamples
        self.scatter_normz = scatter_normz
        self.est_marginal_sp = est_marginal_sp
        self.dist_type = dist_type

        self.q = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, 1, bias=qkv_bias)

        self.norm = LayerNorm2d(dim)
        self.vweight = vweight
        self.out_weight = out_weight

        self.attn_drop = nn.Dropout(attn_drop)
        head_dim = self.qk_dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

    def forward(self, x, affinity_matrix=None, num_spixels=None,
                sims=None, indices=None, labels=None, mask=None):


        # -- compute the q,k,v --
        B, C, H, W = x.shape
        x = self.norm(x)
        q, k, v = self.q(x), self.k(x), self.v(x)

        # -- unpack superpixel info --
        if (affinity_matrix is None) and (labels is None):
            affinity_matrix, num_spixels = self.est_marginal_sp(x)
        if labels is None:
            sims, indices, labels, mask = unpack_sp(affinity_matrix,self.topk)

        # -- create q,k,v --
        sample_it = lambda qkv,dim: torch.gather(
            qkv.reshape(B, 1, -1, H*W).expand(-1, num_spixels, -1, -1),
            -1, indices.unsqueeze(2).expand(-1, -1, dim, -1)) # B, K, C, topk
        shape_str = 'b k (h c) t -> b k h t c'
        reshape_it = lambda qkv: rearrange(qkv, shape_str, h=self.num_heads)
        Q_sp = reshape_it(sample_it(q,self.qk_dim))
        K_sp = reshape_it(sample_it(k,self.qk_dim))
        V_sp = reshape_it(sample_it(v,self.dim))

        # -- compute attention weights --
        if self.dist_type == "prod":
            attn = (Q_sp @ K_sp.transpose(-2,-1)) * self.scale # b k h topk topk
        elif self.dist_type == "l2":
            _B,_K,_H = Q_sp.shape[:3]
            Q_sp = rearrange(Q_sp,'b k h k1 k2 -> (b k h) k1 k2')
            K_sp = rearrange(K_sp,'b k h k1 k2 -> (b k h) k1 k2')
            attn = -self.scale*th.cdist(Q_sp,K_sp)
            attn = rearrange(attn,'(b k h) k1 k2 -> b k h k1 k2',b=_B,k=_K,h=_H)
        else:
            raise ValueError(f"Uknown dist type [{self.dist_type}]")
        attn = normalize_attention(self.attn_normz,attn,sims,mask,self.normz_nsamples)
        attn = self.attn_drop(attn)

        #
        # -- Weighting by Marginal Probs --
        #

        # -- P(L[j]=s)=sims[s,j] --
        valid_sims = not(sims is None)
        weight = sims[:,:,None,:,None].expand_as(V_sp) if valid_sims else None

        # -- P(L[j] = s) for i ~ j --
        if self.vweight: V_sp = weight *  V_sp
        out = attn @ V_sp # b k h topk c
        # -- P(L[i] = s) for i pixels --
        if self.out_weight: out = weight * out

        #
        # -- Scattering back to img --
        #

        out = rearrange(out, 'b k h t c -> b (h c) (k t)')
        if valid_sims: weight = rearrange(weight,'b k h t c -> b (h c) (k t)')
        out,cnt,wght = scatter_mean(v.reshape(B, self.dim, H*W), -1,
                                    indices.reshape(B, 1, -1).expand(-1, self.dim, -1),
                                    out,weight,self.scatter_normz)
        out = out.reshape(B, C, H, W)

        # -- mask --
        cnt = cnt.reshape(B, C, H, W)
        # if cnt.min().item() == 0:
        #     print(cnt)
        #     exit()
        if self.scatter_normz == "sum2one":
            mask = wght.reshape(B, C, H, W)
            # print(mask)
            # print((mask + (1-mask)).min(),(mask + (1-mask)).min())
        else:
            mask = (cnt>1e-5).type(th.float)
        out = mask * out + (1-mask)*v
        # print("post: ",out.min().item(),out.max().item())
        # print(out.min(),out.max())


        return out

#
#
# -- Gaussian Mixture Model --
#
#

class SpaGmmSampling(nn.Module):
    def __init__(self, attn, est_marginal_sp, nsamples=30, topk=32):
        super().__init__()
        self.attn = attn
        self.nsamples = nsamples
        self.topk = topk
        self.est_marginal_sp = est_marginal_sp

    def forward(self, x):

        # -- unpack --
        nsamples = self.nsamples
        B = x.shape[0]

        # -- precomputing --
        amatrix, num_spixels = self.est_marginal_sp(x)
        sims, indices = torch.topk(amatrix, self.topk, dim=-1) # B, K, topk
        # amatrix = rearrange(amatrix,'b s k -> (b s) k')
        amatrix = rearrange(amatrix,'b s k -> s (b k)') #? check nsp.py
        print("not here.")
        exit()

        # -- get sample --
        labels = th.multinomial(amatrix.T,num_samples=1)
        labels = rearrange(labels,'(b s) k -> b k s',b=B)
        mask = mask_from_labels(labels,indices)

        # -- compute attn --
        out = self.attn(x, None, num_spixels, None,
                        indices=indices, labels=labels, mask=mask)/nsamples

        # -- compute average --
        for _ in range(self.nsamples-1):

            # -- get sample --
            labels = th.multinomial(amatrix.T,num_samples=1)
            labels = rearrange(labels,'(b s) k -> b k s',b=B)
            mask = mask_from_labels(labels,indices)

            # -- compute attn --
            out += self.attn(x, None, num_spixels, None,
                             indices=indices, labels=labels, mask=mask)/nsamples
        return out

#
#
# -- Sampling Generic Model --
#
#

class SpaSampling(nn.Module):
    def __init__(self, attn, get_sp_sample, nsamples=30, topk=32):
        super().__init__()
        self.attn = attn
        self.nsamples = nsamples
        self.topk = topk
        self.get_sp_sample = get_sp_sample

    def forward(self, x):

        # -- unpack --
        nsamples = self.nsamples
        B = x.shape[0]

        # -- compute sp --
        # amatrix, num_spixels = self.est_marginal_sp(x)
        # indices, labels, mask = self.est_marginal_sp(x)
        indices, labels, mask = self.get_sp_sample(x)

        # # -- prepare --
        # sims, indices = torch.topk(amatrix, self.topk, dim=-1) # B, K, topk

        # # -- sampling labels --
        # amatrix = rearrange(amatrix,'b s k -> (b s) k')
        # labels = th.multinomial(amatrix.T,num_samples=1)
        # labels = rearrange(labels,'(b s) k -> b k s',b=B)
        # mask = mask_from_labels(labels,indices)
        out = self.attn(x, None, None, None,
                        indices=indices, labels=labels, mask=mask)/nsamples

        # -- compute average --
        for _ in range(self.nsamples-1):

            # -- samples --
            # labels = th.multinomial(amatrix.T,num_samples=1)
            # labels = rearrange(labels,'(b s) k -> b k s',b=B)
            # mask = mask_from_labels(labels,indices)
            indices, labels, mask = self.get_sp_sample(x)
            out += self.attn(x, None, None, None,
                             indices=indices, labels=labels, mask=mask)/nsamples
        return out

