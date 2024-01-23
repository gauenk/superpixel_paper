
# -- import torch --
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# -- basic --
import math
from einops import rearrange
from easydict import EasyDict as edict

class SimulateBass():

    def __init__(self,sp_stride,topk):
        self.sp_stride = sp_stride
        self.topk = topk

    def __call__(self,x):

        # -- sample --
        seed = th.rand(1).item()
        labels = run_bass_sp(x,seed)

        # -- expand to sims --
        sims = expand_labels(labels,self.sp_stride)

        # ? -- ?
        _, indices = torch.topk(sims, topk, dim=-1) # B, K, topk

        mask_from_labels(labels,indices)

        return indices, labels, mask


# -- misc --
# Global.repeat=False
# count = 0
# for Global.IMAGE1 in image_files:
#     torch.cuda.empty_cache()
#     count = count + 1
#     Global.initVariables()
#     Global.Beta_P = (args.potts-2.7) + Global.Beta_P
#     Global.K_C = args.sp
#     Global.csv_file=Global.IMAGE1[Global.IMAGE1.rfind("/")+1:][:-4]
#     Global.repeat=False
#     print(Global.csv_file)
#     SuperPixelsSplitMerge()

def mask_from_labels(labels,indices):
    K,topk = indices.shape[1:3] # number of superpixels
    lids = th.gater(labels,sp_locs)
    # lids = th.arange(K).to(indices.device).reshape(1,K,1).expand_as(indices)
    labels = th.gather(labels.expand(-1,K,-1),-1,indices)
    mask = 1.*(labels == lids)
    mask = mask.reshape(1,K,1,topk,1)
    return mask


def run_bass_sp(x,seed):
    return labels

