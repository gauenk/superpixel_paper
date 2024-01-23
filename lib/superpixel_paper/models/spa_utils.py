# -- import torch --
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# -- basic --
import math
from einops import rearrange
from easydict import EasyDict as edict

# -- superpixel --
from superpixel_paper.est_attn_normz import EffNormzFunction
from spin.models.pair_wise_distance import PairwiseDistFunction
from natten import NeighborhoodAttention2D

def normalize_attention(version,attn,sims,mask,nsamples):
    # print(version)
    # exit()
    if version == "sample":
        # print("nsamples: ",nsamples)
        # exit()
        # c = (attn).max().item()
        # attn = th.exp(attn - c + th.log(eps+th.sum(th.exp(attn-c))))
        # attn = th.exp(attn)
        # eps = 1e-10
        c = (attn).max().item()
        # print(attn.min().item(),attn.max().item())

        # -- [cuda kernel; not needed anymore i think...] --
        # normz = estimate_normz(th.exp(attn-c),sims,nsamples)

        sims = sims[...,None].expand(-1,-1,-1,nsamples)
        samples = th.bernoulli(sims)
        # normz = th.sum(mask*th.exp(attn-c))
        exit()


        attn = th.exp(attn-c-th.log(normz))
        # print(attn.min().item(),attn.max().item())
        # # eps = 1e-6
        # # th.log(normz+eps)
        # attn = attn * normz
        # elif version in ["mle","mle_z"]:
    elif version  == "mle":

        # -- mask --
        mask = mask[...,None,:,0]

        # -- attn --
        eps = 1e-6
        c = (mask*attn).max().item()
        attn = th.exp(mask*attn - c - th.log(eps+th.sum(mask*th.exp(attn-c))))

        # -- softmax --
        # attn = nz_mask*(attn).softmax(dim=-1)
        # print(attn.min().item(),attn.max().item())
        if th.any(th.isnan(attn)):
            print(mask.sum(-1),mask.shape)
            print(attn.min().item(),attn.max().item())
            print("nan!")
            exit()
        # print("yo!")
        # exit()

    elif version == "mle_z":
        # -- attn --
        mask = mask[...,None,:,0]
        eps = 1e-6
        c = (mask*attn).max().item()
        attn = th.exp(attn - c - th.log(eps+th.sum(mask*th.exp(attn-c))))
        # eps = 1e-6
        # attn = th.exp(attn)
        # attn = attn / (eps+th.sum(mask*attn,-1,keepdim=True))
        # print("[min,max]: ",attn.min().item(),attn.max().item())
        # exit()
    elif version == "softmax":
        attn = attn.softmax(dim=-1)
    else:
        raise ValueError("Uknown attention normz version [%s]"%str(version))
    return attn

def estimate_normz(attn,sims,nsamples):
    samples = EffNormzFunction.sample(sims,nsamples)
    normz = EffNormzFunction.apply(attn,samples)
    if th.any(normz==0):
        print(attn)
        print(normz)
    assert th.all(normz>0).item(),"Must be nz"
    return normz

def unpack_sp(affinity_matrix,topk):

    # -- sample superpixel --
    sims, indices = torch.topk(affinity_matrix, topk, dim=-1) # B, K, topk

    # -- get mask --
    _,labels = torch.topk(affinity_matrix.detach(),1,dim=-2)
    mask = mask_from_labels(labels,indices)

    return sims,indices,labels,mask

def mask_from_labels(labels,indices):
    K,topk = indices.shape[1:3] # number of superpixels
    lids = th.arange(K).to(indices.device).reshape(1,K,1).expand_as(indices)
    labels = th.gather(labels.expand(-1,K,-1),-1,indices)
    mask = 1.*(labels == lids)
    mask = mask.reshape(1,K,1,topk,1)
    return mask

def weighted_scatter_mean(tgt, weight, mask, dim, indices, src ,normz=False):
    count = torch.ones_like(tgt)
    new_src = torch.scatter_add(tgt, dim, indices, src)
    if not(normz is False) and not(normz is None):
        if (normz is True) or (normz == "default"):
            new_count = torch.scatter_add(count, dim, indices, weight)
            new_src /= new_count
        elif (normz == "ones"):
            ones = th.ones_like(weight)
            new_count = torch.scatter_add(count, dim, indices, ones)
            new_src /= new_count
            # print(new_src)
        elif (normz == "mask"):
            new_count = torch.scatter_add(count, dim, indices, mask)
            new_src /= new_count
        else:
            raise ValueError("Uknown normalization.")
    return new_src

def scatter_mean(tgt, dim, indices, src, weights=None, divide=True):
    new_src = torch.scatter_add(tgt, dim, indices, src)
    count = torch.zeros_like(tgt)
    new_count = torch.scatter_add(count, dim, indices, torch.ones_like(src))
    if divide == "ones":
        eps = 1e-6
        count = torch.ones_like(tgt)
        new_count = torch.scatter_add(count, dim, indices, torch.ones_like(src))
        new_src = new_src / (eps+new_count)
    elif divide == "weights":
        eps = 1e-6
        count = torch.zeros_like(tgt)
        new_count = torch.scatter_add(count, dim, indices, weights)
        # print("pre: ",new_src.min().item(),new_src.max().item())
        # print(new_count)
        # print(new_count.min(),new_count.max())
        # print("pre: ",new_src.min().item(),new_src.max().item())
        # print(new_count.shape)
        # print(th.histogram(new_count[0,0].cpu()))
        new_src = new_src / (eps+new_count)
        # print("post: ",new_src.min().item(),new_src.max().item())
        # exit()
        # print(new_src.min().item(),new_src.max().item())

    # -- provide weights --
    wght = None
    if divide == "sum2one":
        wght = torch.zeros_like(tgt)
        wght = torch.scatter_add(wght, dim, indices, weights)

    return new_src,new_count,wght
