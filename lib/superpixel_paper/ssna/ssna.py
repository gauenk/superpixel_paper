
import torch
import torch as th
from einops.layers.torch import Rearrange


import torch
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_
from einops import rearrange,repeat
from natten.functional import natten2dav, natten2dqkrpb

# -- ssna --
from superpixel_paper.nat.nat_spin import NeighAttnMat
from superpixel_paper.nat.nat_spin import NeighAttnAgg
from ..sna.sna_attn import NeighSuperpixelAttn
from .ssna_attn import SoftNeighSuperpixelAttn
from .ssna_agg import SoftNeighSuperpixelAgg
from .attn_reweight import AttnReweight
from .attn_reweight_v2 import AttnReweightV2

class SoftSuperpixelNeighborhoodAttention(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(self,dim,num_heads,kernel_size,
                 dilation=1,bias=False,qk_scale=None,
                 attn_drop=0.0,proj_drop=0.0,mask_labels=False,
                 use_proj=True,use_weights=True,
                 qk_layer=None,v_layer=None,proj_layer=None,
                 learn_attn_scale=False,detach_sims=False,
                 detach_learn_attn=False,attn_rw_version="v0"):
        super().__init__()

        # -- superpixels --
        self.qk_scale = qk_scale
        self.mask_labels = mask_labels
        self.detach_sims = detach_sims

        # -- scaling --
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.use_weights = use_weights

        # -- neighborhood attn --
        bias = False
        # self.nat_attn = SoftNeighSuperpixelAttn(dim=dim,
        #                                        kernel_size=kernel_size,
        #                                        num_heads=num_heads,
        #                                         qk_bias=bias,
        #                                         use_weights=use_weights)
        # self.nat_attn = NeighSuperpixelAttn(dim=dim,kernel_size=kernel_size,
        #                                     num_heads=num_heads,
        #                                     qk_bias=bias,use_weights=use_weights,
        #                                     qk_layer=qk_layer,qk_scale=qk_scale,
        #                                     learn_attn_scale=learn_attn_scale)
        self.nat_attn = NeighAttnMat(dim=dim, kernel_size=kernel_size,
                                     dilation=1, num_heads=num_heads, bias=bias,
                                     qk_bias=bias, qk_scale=qk_scale,
                                     learn_attn_scale=learn_attn_scale,
                                     detach_learn_attn=detach_learn_attn)
        self.attn_rw_version = attn_rw_version
        if attn_rw_version == "v0":
            self.attn_rw = AttnReweight()
        elif attn_rw_version == "v1":
            self.attn_rw = AttnReweightV2()
        else:
            raise KeyError(f"Uknown attention re-weight function [{attn_rw_version}]")
        assert (use_weights is True)
        self.nat_agg = NeighAttnAgg(dim=dim,num_heads=num_heads,
                                    kernel_size=kernel_size,
                                    v_bias=bias,use_proj=use_proj)
        # self.nat_agg = SoftNeighSuperpixelAgg(dim=dim,num_heads=num_heads,
        #                                       kernel_size=kernel_size,
        #                                       v_bias=bias,use_proj=use_proj,
        #                                       use_weights=use_weights,
        #                                       v_layer=v_layer,proj_layer=proj_layer)

    def forward(self, x, sims, state=None):

        # -- unpack superpixel info --
        if self.detach_sims:
            sims = sims.detach()

        # -- reshape --
        x = x.permute(0,2,3,1) # b f h w -> b h w f

        # -- attn map --
        attn = self.nat_attn(x)
        # print(attn[0,0,0,0,:])
        # print(attn[0,0,1,1,:])
        # exit()

        # -- reweight attn map with probs --
        if self.mask_labels:
            attn = attn.softmax(-1)
        else:

            # -- indices for access --
            sims = sims.contiguous()
            device = sims.device
            B,H,W,F = x.shape
            sH,sW = sims.shape[-2:]
            sinds = get_indices(H,W,sH,sW,device)

            # print("-"*10)
            # print(attn[0,0,0,0,:])
            # print(attn[0,0,1,1,:])
            # print("-"*10)
            # print((attn.softmax(-1))[0,0,0,0,:])
            # print((attn.softmax(-1))[0,0,1,1,:])

            # -- reweight with P(L_j = s) --
            attn = self.attn_rw(attn,sims,sinds)

            # print("-"*10)
            # print(attn[0,0,0,0])
            # print(attn[0,0,1,1])
            # exit()

        # -- aggregate --
        x = self.nat_agg(x,attn)

        # -- prepare --
        x = x.permute(0,3,1,2)#.clone() # b h w f -> b f h w
        return x



def get_indices(H,W,sH,sW,device):
    sHW = sH*sW
    labels = th.arange(sHW, device=device).reshape(1, 1, sH, sW).float()
    interp = th.nn.functional.interpolate
    labels = interp(labels, size=(H, W), mode="nearest").long()[0,0]
    labels = th.stack([labels/sW,labels%sW],-1).long()
    return labels

