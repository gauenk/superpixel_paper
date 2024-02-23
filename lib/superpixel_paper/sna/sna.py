
import torch
import torch as th
from einops.layers.torch import Rearrange


import torch
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_
from einops import rearrange,repeat

from natten.functional import natten2dav, natten2dqkrpb

# from .functional import natten2dav, natten2dqkrpb
# from natten import NeighAttnMat
from .sna_attn import NeighSuperpixelAttn
from .sna_agg import NeighSuperpixelAgg


class NeighborhoodSuperpixelAttention(nn.Module):
    """

    Superpixel Neighborhood Attention

    """

    def __init__(self,dim,num_heads,kernel_size,
                 dilation=1,bias=False,qkv_bias=False,qk_scale=None,
                 attn_drop=0.0,proj_drop=0.0,mask_labels=False,
                 use_proj=True,use_weights=True,
                 qk_layer=None,v_layer=None,proj_layer=None,
                 learn_attn_scale=False):
        super().__init__()

        # -- superpixels --
        self.mask_labels = mask_labels

        # -- scaling --
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.use_weights = use_weights

        # -- neighborhood attn --
        bias = False
        self.nat_mat = NeighSuperpixelAttn(dim=dim,
                                           kernel_size=kernel_size,
                                           num_heads=num_heads,
                                           qk_bias=bias,
                                           use_weights=use_weights,
                                           qk_layer=qk_layer,
                                           learn_attn_scale=learn_attn_scale,
                                           qk_scale=qk_scale)
        self.nat_agg = NeighSuperpixelAgg(dim=dim,num_heads=num_heads,
                                          kernel_size=kernel_size,
                                          v_bias=bias,use_proj=use_proj,
                                          use_weights=use_weights,
                                          v_layer=v_layer,
                                          proj_layer=proj_layer)

    def forward(self, x, labels):

        # -- unpack superpixel info --
        if self.mask_labels:
            labels = th.zeros_like(labels)

        # -- reshape --
        x = x.permute(0,2,3,1) # b f h w -> b h w f

        # -- attn map --
        attn = self.nat_mat(x,labels)

        # -- rescale attn --
        attn = attn.softmax(-1)

        # mask = (attn < 1e-8).float().detach()
        # attn = mask * attn + (1-mask)*1e-4
        # print(attn[0,0,5,:20].max(-1))
        # print(attn[0,0,32,32:64].max(-1))
        # print(attn.shape)
        # exit()
        # print("a:",th.any(th.isnan(x)))
        # print(th.any(th.isnan(attn)))
        # exit()

        # print(attn.shape)
        # exit()

        # -- innter product --
        x = self.nat_agg(x,attn)
        # print("b:",th.any(th.isnan(x)))
        # exit()

        # -- prepare --
        x = x.permute(0,3,1,2).clone() # b h w f -> b f h w
        return x

    def masked_softmax(self,attn,sims,mask,version):
        raise NotImplementedError("")

        # -- create mask --
        if version in ["mask"]:
            eps = 1e-6
            c = (mask*attn).max().item()
            attn = th.exp(mask*attn - c - th.log(eps+th.sum(mask*th.exp(attn-c))))
        else:
            raise KeyError("")
        return attn


def unpack_local_sp(A_sp,ksize):

    # # -- sample superpixel --
    # sims = unfold_like_nat(A_sp,ksize)
    # # sims, indices = torch.topk(affinity_matrix, topk, dim=-1) # B, K, topk

    # -- get mask --
    _,labels = torch.topk(A_sp.detach(),1,dim=-2)
    # labels = unfold_like_nat(labels,ksize)
    # mask = mask_from_labels(labels)

    return sims,labels



#
#
# -- Gaussian Mixture Model --
#
#

class SnaGmmSampling(nn.Module):
    def __init__(self, attn, est_marginal_sp, nsamples=10):
        super().__init__()
        self.attn = attn
        self.nsamples = nsamples
        self.est_marginal_sp = est_marginal_sp

    def forward(self, x):

        # -- unpack --
        nsamples = self.nsamples
        B = x.shape[0]
        H = x.shape[-2]

        # -- precomputing --
        sims, num_spixels = self.est_marginal_sp(x)
        sims = rearrange(sims,'b s k -> (b k) s')

        # -- get sample --
        labels = th.multinomial(sims,num_samples=1)
        labels = rearrange(labels,'(b h w) 1 -> b h w',b=B,h=H)
        out = self.attn(x, labels)/nsamples

        # -- compute average --
        for _ in range(self.nsamples-1):

            # -- get sample --
            labels = th.multinomial(sims,num_samples=1)
            labels = rearrange(labels,'(b h w) 1 -> b h w',b=B,h=H)
            out += self.attn(x, labels)/nsamples
        return out

