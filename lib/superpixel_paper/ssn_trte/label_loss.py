import torch
import torch as th
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import stnls
from einops import rearrange
from easydict import EasyDict as edict
from superpixel_paper.utils import append_grid
from torch.nn.functional import one_hot

class SuperpixelLoss(nn.Module):

    def __init__(self,loss_type):
        super().__init__()
        self.loss_type = loss_type
        assert self.loss_type in ["cross","mse"]

    def forward(self,labels,sims):
        assert self.loss_type in ["cross","mse"]

        # -- alloc [compact loss] --
        B,F,H,W = labels.shape
        zeros = th.zeros_like(labels[:,0])

        # -- normalize across #sp for each pixel --
        # sims.shape = B, NumSuperpixels, NumPixels
        sims_nmz = sims.transpose(-1,-2)
        sims_nmz = sims_nmz / sims_nmz.sum(-2,keepdim=True)

        # -- prepare labels --
        labels = labels.flatten(-2,-1)
        # print(labels.shape)
        if self.loss_type == "cross":
            labels = one_hot(labels[:,0].long())*1.
        else:
            labels = rearrange(labels,'b c hw -> b hw c')

        # -- compute "superpixel loss" --
        labels_sp = sims_nmz @ (sims @ labels)
        if self.loss_type == "cross":
            cross = torch.nn.functional.cross_entropy
            labels = rearrange(labels,'b hw c -> (b hw) c')
            labels_sp = rearrange(labels_sp,'b hw c -> (b hw) c')
            # print(labels_sp.shape,labels.shape)
            sp_loss = cross(labels_sp,labels)
            # print(sp_loss)
            # print(labels_sp[100:103])
            # print(labels[100:103])
            # exit()
        elif self.loss_type == "mse":
            # print(labels.shape,labels_sp.shape)
            sp_loss = th.mean((labels - labels_sp)**2)
            # print(sp_loss)
            # exit()

        # -- compute "compact loss" --
        inds = sims.argmax(-2).detach() # hard association
        ixy = append_grid(zeros[:,None,None],1)[:,0,1:]
        ix = ixy[:,0].flatten(-2,-1)[:,:,None]/H
        iy = ixy[:,1].flatten(-2,-1)[:,:,None]/W

        ix_proj = (sims @ ix)[...,0]
        ix_sp = th.gather(ix_proj,-1,inds)

        iy_proj = (sims @ iy)[...,0]
        iy_sp = th.gather(iy_proj,-1,inds)

        compact_loss_x = th.mean((ix_sp - ix)**2)
        compact_loss_y = th.mean((iy_sp - iy)**2)
        compact_loss = (compact_loss_x + compact_loss_y)/2.
        # print(compact_loss)

        # -- final loss --
        lamb = 1e-6
        loss = sp_loss + lamb * compact_loss
        # print(loss)
        # print(sims)
        if th.isnan(loss):
            print(sims)
            exit()

        return loss

