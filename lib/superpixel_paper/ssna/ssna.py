
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
from ..sna.sna_attn import NeighSuperpixelAttn
from .ssna_attn import SoftNeighSuperpixelAttn
from .ssna_agg import SoftNeighSuperpixelAgg
from .attn_reweight import AttnReweight

class SoftSuperpixelNeighborhoodAttention(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(self,dim,num_heads,kernel_size,
                 dilation=1,bias=False,qkv_bias=False,qk_scale=None,
                 attn_drop=0.0,proj_drop=0.0,mask_labels=False,
                 use_proj=True,use_weights=True,
                 qk_layer=None,v_layer=None,proj_layer=None,
                 learn_attn_scale=False):
        super().__init__()

        # -- superpixels --
        self.qk_scale = qk_scale
        self.mask_labels = mask_labels

        # -- scaling --
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        # self.qk_scale = qk_scale or self.head_dim**-0.5
        self.use_weights = use_weights

        # -- neighborhood attn --
        bias = False
        # self.nat_attn = SoftNeighSuperpixelAttn(dim=dim,
        #                                        kernel_size=kernel_size,
        #                                        num_heads=num_heads,
        #                                         qk_bias=bias,
        #                                         use_weights=use_weights)
        self.nat_attn = NeighSuperpixelAttn(dim=dim,
                                            kernel_size=kernel_size,
                                            num_heads=num_heads,
                                            qk_bias=bias,use_weights=use_weights,
                                            qk_layer=qk_layer,qk_scale=qk_scale,
                                            learn_attn_scale=learn_attn_scale)
        self.reweight_version = "exp"
        # if self.reweight_version == "v0":
        #     self.attn_rw = AttnReweight()
        # else:
        #     self.attn_rw = AttnReweightExp()
        self.attn_rw = AttnReweight()
        self.nat_agg = SoftNeighSuperpixelAgg(dim=dim,num_heads=num_heads,
                                              kernel_size=kernel_size,
                                              v_bias=bias,use_proj=use_proj,
                                              use_weights=use_weights,
                                              v_layer=v_layer,proj_layer=proj_layer)
    def forward(self, x, sims, state=None):

        # -- unpack superpixel info --
        if self.mask_labels:
            sims = th.ones_like(sims).contiguous().detach()/9.
            # TODO: dummy data isn't "1/9." since not all have 9.
            # print("hi.")
            # exit()
        else:
            sims = sims.contiguous()

        # -- reshape --
        x = x.permute(0,2,3,1) # b f h w -> b h w f

        # -- indices for access --
        device = sims.device
        B,H,W,F = x.shape
        sH,sW = sims.shape[-2:]
        sinds = self.get_indices(H,W,sH,sW,device)
        # print(x.shape)
        # print(sims.shape)
        # print(sinds)
        # print(sims.sum((-2,-1)).mean())
        # print(sims.sum((-2,-1)).max())
        # print(sims.sum((-2,-1)).min())
        # print(sims.shape)
        # exit()


        # -- attn map --
        eps = 1e-12
        # log_sims = th.log(sims+eps)
        # ones_sims = th.ones_like(sims)
        # attn = self.nat_attn(x,ones_sims,sinds)
        labels = th.zeros((B,H,W),device=device,dtype=th.long)
        attn = self.nat_attn(x,labels)

        # attn = th.mean(attn,2) # all the same
        # print(attn.shape)
        # print(attn[0,0,:,4,4,:])
        # print(attn[0,0,:,14,14,:])
        # print(attn[0,0,:,15,15,:])
        # # print(attn.shape,x.shape,sims.shape)

        # print("attn.shape: ",attn.shape)

        # -- reweight with probs --
        if self.mask_labels:
            attn = attn.softmax(-1)
            attn = attn[:,:,None].repeat(1,1,9,1,1,1)
        else:
            attn = self.attn_rw(attn,sims,sinds)
        # attn = (self.qk_scale * attn).softmax(-1)
        # print("[2] attn.shape: ",attn.shape)

        # exit()
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


    def get_indices(self,H,W,sH,sW,device):
        sHW = sH*sW
        labels = th.arange(sHW, device=device).reshape(1, 1, sH, sW).float()
        interp = th.nn.functional.interpolate
        labels = interp(labels, size=(H, W), mode="nearest").long()[0,0]
        labels = th.stack([labels/sW,labels%sW],-1).long()
        return labels

