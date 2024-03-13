
import torch
import torch as th
from einops.layers.torch import Rearrange


import torch
from torch import nn
from torch.nn.functional import pad,one_hot
from torch.nn.init import trunc_normal_
from einops import rearrange,repeat

from natten.functional import natten2dav, natten2dqkrpb

# from .functional import natten2dav, natten2dqkrpb
# from natten import NeighAttnMat
from superpixel_paper.nat.nat_spin import NeighAttnMat,NeighAttnAgg
from .sna_attn import NeighSuperpixelAttn
from .sna_agg import NeighSuperpixelAgg
from superpixel_paper.ssna.attn_reweight import AttnReweight
from superpixel_paper.ssna.ssna_gather_sims import SsnaGatherSims

class SuperpixelNeighborhoodAttention(nn.Module):
    """

    Superpixel Neighborhood Attention

    """

    def __init__(self,dim,num_heads,kernel_size,
                 dilation=1,bias=False,qkv_bias=False,qk_scale=None,
                 attn_drop=0.0,proj_drop=0.0,mask_labels=False,
                 use_proj=True,use_weights=True,
                 qk_layer=None,v_layer=None,proj_layer=None,
                 learn_attn_scale=False,detach_sims=False,
                 detach_learn_attn=False):
        super().__init__()

        # -- superpixels --
        self.mask_labels = mask_labels
        self.detach_sims = detach_sims

        # -- scaling --
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.use_weights = use_weights
        self.detach_sims = detach_sims
        self.detach_learn_attn = detach_learn_attn

        # -- neighborhood attn --
        bias = False
        self.nat_attn = NeighAttnMat(dim=dim, kernel_size=kernel_size,
                                     dilation=1, num_heads=num_heads, bias=bias,
                                     qk_bias=bias, qk_scale=qk_scale,
                                     learn_attn_scale=learn_attn_scale,
                                     detach_learn_attn=detach_learn_attn)
        self.attn_rw = AttnReweight()
        self.nat_agg = NeighAttnAgg(dim=dim,num_heads=num_heads,
                                    kernel_size=kernel_size,
                                    v_bias=bias,use_proj=use_proj)

        # -- old --
        # self.nat_mat = NeighAttnMat(dim=dim, kernel_size=nat_ksize,
        #                             dilation=1, num_heads=heads, bias=bias,
        #                             qkv_bias=bias, qk_scale=qk_scale,
        #                             learn_attn_scale=learn_attn_scale)
        # if self.mask_labels:
        #     self.nat_mat = NeighAttnMat(dim=dim, kernel_size=nat_ksize,
        #                                 dilation=1, num_heads=heads, bias=bias,
        #                                 qkv_bias=bias, qk_scale=qk_scale,
        #                                 learn_attn_scale=learn_attn_scale)
        # else:
        #     self.nat_mat = NeighSuperpixelAttn(dim=dim, kernel_size=kernel_size,
        #                                        num_heads=num_heads, qk_bias=bias,
        #                                        use_weights=use_weights,
        #                                        qk_layer=qk_layer,qk_scale=qk_scale,
        #                                        learn_attn_scale=learn_attn_scale)
        # self.nat_agg = NeighSuperpixelAgg(dim=dim,num_heads=num_heads,
        #                                   kernel_size=kernel_size,
        #                                   v_bias=bias,use_proj=use_proj,
        #                                   use_weights=use_weights,
        #                                   v_layer=v_layer,
        #                                   proj_layer=proj_layer)

    def forward(self, x, sims):

        # -- unpack superpixel info --
        if self.detach_sims:
            sims = sims.detach()

        # -- reshape --
        x = x.permute(0,2,3,1) # b f h w -> b h w f

        # -- attn map --
        attn = self.nat_attn(x)

        # -- rescale attn --
        if self.mask_labels:
            attn = attn.softmax(-1)
        else:

            # -- indices for access --
            device = sims.device
            B,H,W,F = x.shape
            sH,sW = sims.shape[-2:]
            sinds = get_indices(H,W,sH,sW,device)

            # -- from probs to inds --
            inds = sims.view(*sims.shape[:-2],-1).argmax(-1)
            binary = one_hot(inds,sH*sW)
            binary = binary.view(sims.shape).type(sims.dtype)

            # -- reweight with P(L_j = s) --
            attn = self.attn_rw(attn,binary,sinds)

            # # -- reweight with P(L_i = s) --
            # gather_sims = SsnaGatherSims()
            # pi = gather_sims(binary,sinds)
            # pi = rearrange(pi,'b h w ni -> b 1 ni h w 1')
            # attn = th.sum(pi * attn,2).contiguous()

        # -- innter product --
        x = self.nat_agg(x,attn)

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



def get_indices(H,W,sH,sW,device):
    sHW = sH*sW
    labels = th.arange(sHW, device=device).reshape(1, 1, sH, sW).float()
    interp = th.nn.functional.interpolate
    labels = interp(labels, size=(H, W), mode="nearest").long()[0,0]
    labels = th.stack([labels/sW,labels%sW],-1).long()
    return labels



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

