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


class SPIntraAttModule(nn.Module):
    def __init__(self, dim, num_heads, qk_dim, topk=32,
                 qkv_bias=False, qk_scale=None, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.num_heads = num_heads

        self.topk = topk

        self.q = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, 1, bias=qkv_bias)

        self.norm = LayerNorm2d(dim)

        self.attn_drop = nn.Dropout(attn_drop)
        head_dim = self.qk_dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

    def forward(self, x, affinity_matrix, num_spixels):
        B, C, H, W = x.shape

        # calculate v
        x = self.norm(x)
        q, k, v = self.q(x), self.k(x), self.v(x)

        # superpixels' pixel selection
        _misc, indices = torch.topk(affinity_matrix, self.topk, dim=-1) # B, K, topk
        # print(self.topk)
        # print(_misc)
        # exit()

        q_sp_pixels = torch.gather(q.reshape(B, 1, -1, H*W).expand(-1, num_spixels, -1, -1), -1, indices.unsqueeze(2).expand(-1, -1, self.qk_dim, -1)) # B, K, C, topk
        k_sp_pixels = torch.gather(k.reshape(B, 1, -1, H*W).expand(-1, num_spixels, -1, -1), -1, indices.unsqueeze(2).expand(-1, -1, self.qk_dim, -1)) # B, K, C, topk
        v_sp_pixels = torch.gather(v.reshape(B, 1, -1, H*W).expand(-1, num_spixels, -1, -1), -1, indices.unsqueeze(2).expand(-1, -1, self.dim, -1)) # B, K, C, topk

        q_sp_pixels, k_sp_pixels, v_sp_pixels = \
            map(lambda t: rearrange(t, 'b k (h c) t -> b k h t c', h=self.num_heads),
                (q_sp_pixels, k_sp_pixels, v_sp_pixels)) # b k topk c

        # similarity
        attn = (q_sp_pixels @ k_sp_pixels.transpose(-2,-1)) * self.scale # b k h topk topk
        # attn = F.normalize(q_sp_pixels, dim=-1) @ F.normalize(k_sp_pixels, dim=-1).transpose(-2,-1) # b k h topk topk
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v_sp_pixels # b k h topk c
        out = rearrange(out, 'b k h t c -> b (h c) (k t)')

        out = scatter_mean(v.reshape(B, self.dim, H*W), -1,
                           indices.reshape(B, 1, -1).expand(-1, self.dim, -1), out)
        out = out.reshape(B, C, H, W)

        return out

class SPIntraAttModuleV2(nn.Module):
    def __init__(self, dim, num_heads, qk_dim, topk=32,
                 qkv_bias=False, qk_scale=None, attn_drop=0.,
                 normz=False,kweight=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.num_heads = num_heads

        self.topk = topk

        self.q = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, 1, bias=qkv_bias)

        self.norm = LayerNorm2d(dim)
        self.normz = normz
        self.kweight = kweight

        self.attn_drop = nn.Dropout(attn_drop)
        head_dim = self.qk_dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

    def forward(self, x, affinity_matrix, num_spixels):
        B, C, H, W = x.shape

        # calculate v
        x = self.norm(x)
        q, k, v = self.q(x), self.k(x), self.v(x)
        # v = self.v(x)

        # superpixels' pixel selection; K = # of superpixels
        # print(affinity_matrix[0][0])
        sims, indices = torch.topk(affinity_matrix, self.topk, dim=-1) # B, K, topk
        # sims = th.softmax(sims,-1) # B K, topk

        sample_it = lambda qkv,dim: torch.gather(qkv.reshape(B, 1, -1, H*W).expand(-1, num_spixels, -1, -1), -1, indices.unsqueeze(2).expand(-1, -1, dim, -1)) # B, K, C, topk
        shape_str = 'b k (h c) t -> b k h t c'
        reshape_it = lambda qkv: rearrange(qkv, shape_str, h=self.num_heads)
        q_sp_pixels = reshape_it(sample_it(q,self.qk_dim))
        k_sp_pixels = reshape_it(sample_it(k,self.qk_dim))
        v_sp_pixels = reshape_it(sample_it(v,self.dim))

        # -- similarity --
        attn = (q_sp_pixels @ k_sp_pixels.transpose(-2,-1)) * self.scale # b k h topk topk
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weight = sims[:,:,None,:,None].expand_as(v_sp_pixels)
        if self.kweight:
            v_sp_pixels = weight *  v_sp_pixels
        out = attn @ v_sp_pixels # b k h topk c
        # weight = sims[:,:,None,:,None].expand_as(out)
        out = rearrange(out, 'b k h t c -> b (h c) (k t)')
        weight = rearrange(weight,'b k h t c -> b (h c) (k t)')
        out = weight*out
        # weight = th.ones_like(weight)
        out = weighted_scatter_mean(v.reshape(B, self.dim, H*W), weight, None, -1,
                                    indices.reshape(B, 1, -1).expand(-1, self.dim, -1),
                                    out,normz=self.normz)
        out = out.reshape(B, C, H, W)

        return out

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
        elif (normz == "mask"):
            new_count = torch.scatter_add(count, dim, indices, mask)
            new_src /= new_count
        else:
            raise ValueError("Uknown normalization.")
    return new_src

def scatter_mean(tgt, dim, indices, src):
    count = torch.ones_like(tgt)
    new_src = torch.scatter_add(tgt, dim, indices, src)
    new_count = torch.scatter_add(count, dim, indices, torch.ones_like(src))
    new_src /= new_count

    return new_src


class SPIntraAttModuleV3(nn.Module):
    def __init__(self, dim, num_heads, qk_dim, topk=32,
                 qkv_bias=False, qk_scale=None, attn_drop=0.,
                 normz=False,kweight=True,out_weight=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.num_heads = num_heads

        self.topk = topk

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

    def forward(self, x, affinity_matrix, num_spixels,
                sims=None, indices=None, labels=None):
        B, C, H, W = x.shape

        # calculate v
        x = self.norm(x)
        q, k, v = self.q(x), self.k(x), self.v(x)
        # v = self.v(x)

        # -- sample superpixel --
        # superpixels' pixel selection; K = # of superpixels
        if (sims is None):
            assert indices is None
            assert labels is None
            sims, indices = torch.topk(affinity_matrix, self.topk, dim=-1) # B, K, topk
            _,labels = torch.topk(affinity_matrix.detach(),1,dim=-2)

        # -- get mask --
        K = indices.shape[1] # number of superpixels
        lids = th.arange(K).to(indices.device).reshape(1,K,1).expand_as(indices)
        labels = th.gather(labels.expand(-1,K,-1),-1,indices)
        mask = 1.*(labels == lids)
        mask = mask.reshape(1,K,1,self.topk,1)

        # -- create q,k,v --
        sample_it = lambda qkv,dim: torch.gather(qkv.reshape(B, 1, -1, H*W).expand(-1, num_spixels, -1, -1), -1, indices.unsqueeze(2).expand(-1, -1, dim, -1)) # B, K, C, topk
        shape_str = 'b k (h c) t -> b k h t c'
        reshape_it = lambda qkv: rearrange(qkv, shape_str, h=self.num_heads)
        q_sp_pixels = mask*reshape_it(sample_it(q,self.qk_dim))
        k_sp_pixels = mask*reshape_it(sample_it(k,self.qk_dim))
        v_sp_pixels = mask*reshape_it(sample_it(v,self.dim))
        # print(q_sp_pixels.shape)
        # print("mask.shape: ",mask.shape)
        # exit()

        # -- similarity --
        attn = (q_sp_pixels @ k_sp_pixels.transpose(-2,-1)) * self.scale # b k h topk topk
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weight = sims[:,:,None,:,None].expand_as(v_sp_pixels)
        mask = mask.expand_as(v_sp_pixels)
        if self.kweight:
            v_sp_pixels = weight *  v_sp_pixels
        out = attn @ v_sp_pixels # b k h topk c
        if self.out_weight:
            out = weight * out
        out = mask * out
        # print(self.kweight,self.out_weight,self.normz)
        # print(out.shape,mask.shape,weight.shape)
        # exit()
        # weight = sims[:,:,None,:,None].expand_as(out)
        out = rearrange(out, 'b k h t c -> b (h c) (k t)')
        mask = rearrange(mask, 'b k h t c -> b (h c) (k t)')
        weight = rearrange(weight,'b k h t c -> b (h c) (k t)')
        # weight = th.ones_like(weight)
        out = weighted_scatter_mean(v.reshape(B, self.dim, H*W), weight, mask, -1,
                                    indices.reshape(B, 1, -1).expand(-1, self.dim, -1),
                                    out,normz=self.normz)
        out = out.reshape(B, C, H, W)

        return out

class SPIntraAttModuleV5(nn.Module):
    def __init__(self, exact_attn, nsamples=30, topk=32):
        super().__init__()
        self.exact_attn = exact_attn
        self.nsamples = nsamples
        self.topk = topk

    def forward(self, x, amatrix, num_spixels):

        # -- unpack --
        nsamples = self.nsamples
        B = x.shape[0]

        # -- prepare --
        sims, indices = torch.topk(amatrix, self.topk, dim=-1) # B, K, topk
        amatrix = rearrange(amatrix,'b s k -> (b s) k')

        # -- first sample --
        # _,labels = torch.topk(amatrix.detach(),1,dim=-2)
        labels = th.multinomial(amatrix.T,num_samples=1)
        labels = rearrange(labels,'(b s) k -> b k s',b=B)
        out = self.exact_attn(x, amatrix, num_spixels,
                              sims=sims, indices=indices, labels=labels)/nsamples

        # -- compute average --
        for _ in range(self.nsamples-1):

            # -- samples --
            labels = th.multinomial(amatrix.T,num_samples=1)
            labels = rearrange(labels,'(b s) k -> b k s',b=B)
            out += self.exact_attn(x, amatrix, num_spixels,
                                   sims=sims, indices=indices, labels=labels)/nsamples
        return out

class SPIntraAttModuleV4(nn.Module):
    def __init__(self, dim, num_heads, qk_dim, topk=32,
                 qkv_bias=False, qk_scale=None, attn_drop=0.,
                 normz=False,kweight=True,out_weight=True,nsamples=30):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.num_heads = num_heads
        self.nsamples = nsamples

        self.topk = topk

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

    def estimate_normz(self,attn,sims):
        assert attn.shape[2] == 1,"Num heads is 1 to keep life easy."
        # print("self.nsamples: ",self.nsamples)
        # exit()
        samples = EffNormzFunction.sample(sims,self.nsamples)
        normz = EffNormzFunction.apply(attn,samples)
        if th.any(normz==0):
            print(attn)
            print(normz)
        assert th.all(normz>0).item(),"Must be nz"
        return normz

    def forward(self, x, affinity_matrix, num_spixels):
        B, C, H, W = x.shape

        # calculate v
        x = self.norm(x)
        q, k, v = self.q(x), self.k(x), self.v(x)
        # v = self.v(x)

        # -- sample superpixel --
        # superpixels' pixel selection; K = # of superpixels
        sims, indices = torch.topk(affinity_matrix, self.topk, dim=-1) # B, K, topk

        # -- get mask --
        _,labels = torch.topk(affinity_matrix.detach(),1,dim=-2)
        K = indices.shape[1] # number of superpixels
        lids = th.arange(K).to(indices.device).reshape(1,K,1).expand_as(indices)
        labels = th.gather(labels.expand(-1,K,-1),-1,indices)
        mask = 1.*(labels == lids)
        mask = mask.reshape(1,K,1,self.topk,1)

        # -- create q,k,v --
        sample_it = lambda qkv,dim: torch.gather(qkv.reshape(B, 1, -1, H*W).expand(-1, num_spixels, -1, -1), -1, indices.unsqueeze(2).expand(-1, -1, dim, -1)) # B, K, C, topk
        shape_str = 'b k (h c) t -> b k h t c'
        reshape_it = lambda qkv: rearrange(qkv, shape_str, h=self.num_heads)
        q_sp_pixels = mask*reshape_it(sample_it(q,self.qk_dim))
        k_sp_pixels = mask*reshape_it(sample_it(k,self.qk_dim))
        v_sp_pixels = mask*reshape_it(sample_it(v,self.dim))
        # print(q_sp_pixels.shape)
        # print("mask.shape: ",mask.shape)
        # exit()

        # -- similarity --
        attn = (q_sp_pixels @ k_sp_pixels.transpose(-2,-1)) * self.scale # b k h topk topk
        attn = th.exp(attn)
        normz = self.estimate_normz(attn,sims)
        # normz = th.ones_like(attn)
        # print("attn.shape: ",attn.shape,normz.shape)
        attn = attn*normz
        # attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weight = sims[:,:,None,:,None].expand_as(v_sp_pixels)
        if self.kweight:
            v_sp_pixels = weight *  v_sp_pixels
        out = attn @ v_sp_pixels # b k h topk c
        if self.out_weight:
            out = weight * out
        # print(self.kweight,self.out_weight,self.normz)
        # print(out.shape,mask.shape,weight.shape)
        # exit()
        # weight = sims[:,:,None,:,None].expand_as(out)
        out = rearrange(out, 'b k h t c -> b (h c) (k t)')
        weight = rearrange(weight,'b k h t c -> b (h c) (k t)')
        # weight = th.ones_like(weight)
        out = weighted_scatter_mean(v.reshape(B, self.dim, H*W), weight, None, -1,
                                    indices.reshape(B, 1, -1).expand(-1, self.dim, -1),
                                    out,normz=self.normz)
        out = out.reshape(B, C, H, W)

        return out


