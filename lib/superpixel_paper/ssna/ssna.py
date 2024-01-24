
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
from .ssna_attn import SoftNeighSuperpixelAttn
from .ssna_agg import SoftNeighSuperpixelAgg

class SoftSuperpixelNeighborhoodAttention(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(self,dim,num_heads,kernel_size,
                 vweight=False,oweight=False,attn_normz="mask",
                 dilation=1,bias=False,qkv_bias=False,qk_scale=None,
                 attn_drop=0.0,proj_drop=0.0,mask_labels=False,
                 use_proj=True):
        super().__init__()

        # -- superpixels --
        self.vweight=vweight
        self.oweight=oweight
        self.qk_scale = qk_scale
        self.attn_normz=attn_normz
        self.mask_labels = mask_labels

        # -- scaling --
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.qk_scale = qk_scale or self.head_dim**-0.5

        # -- neighborhood attn --
        bias = False
        self.nat_mat = SoftNeighSuperpixelAttn(dim=dim,
                                               kernel_size=kernel_size,
                                               num_heads=num_heads,
                                               qk_bias=bias)
        self.nat_agg = SoftNeighSuperpixelAgg(dim=dim,num_heads=num_heads,
                                              kernel_size=kernel_size,
                                              v_bias=bias,use_proj=use_proj)

    def forward(self, x, sims):

        # -- unpack superpixel info --
        if self.mask_labels:
            sims = th.ones_like(sims)

        # -- reshape --
        x = x.permute(0,2,3,1) # b f h w -> b h w f

        # -- indices for access --
        H,W = x.shape[1],x.shape[2]
        sH,sW = sims.shape[-2:]
        sinds = get_indices(H,W,sH,sW,sims.device)

        # -- attn map --
        attn = self.nat_mat(x,sims,sinds)
        # print(attn.shape,x.shape,sims.shape)
        attn = (self.qk_scale * attn).softmax(-1)

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
        x = self.nat_agg(x,attn,sims,sinds)

        # -- prepare --
        x = x.permute(0,3,1,2).clone() # b h w f -> b f h w
        return x


def get_indices(H,W,sH,sW,device):
    sHW = sH*sW
    labels = th.arange(sHW, device=device).reshape(1, 1, sH, sW).float()
    interp = th.nn.functional.interpolate
    labels = interp(labels, size=(H, W), mode="nearest").long()[0,0]
    labels = th.stack([labels/sW,labels%sW],-1).long()
    return labels

