import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



def conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

class ResBlockList(nn.Module):

    def __init__(self, nres, n_feats, kernel_size, bn=False, append_noise=False):
        super().__init__()
        self.append_noise = append_noise
        if nres > 0:
            res = []
            for r in range(nres):
                if r == 0 and append_noise is True:
                    res.append(ResBlock(conv, n_feats+1, kernel_size, o_feats=n_feats))
                else:
                    res.append(ResBlock(conv, n_feats, kernel_size))
            if bn:
                res.append(nn.BatchNorm2d(n_feats))
            self.res = nn.Sequential(*res)
        else:
            self.res = nn.Identity()

    def forward(self,vid):
        # B,T = vid.shape[:2]
        # vid = rearrange(vid,'b t c h w -> (b t) c h w')
        vid = self.res(vid)
        # vid = rearrange(vid,'(b t) c h w -> b t c h w',b=B)
        return vid

class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True,
                 act=nn.PReLU(), res_scale=1,  o_feats=None):
        super().__init__()
        if o_feats is None: o_feats = n_feats
        m = []
        m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
        m.append(act)
        m.append(conv(n_feats, o_feats, kernel_size, bias=bias))
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        out = self.body(x).mul(self.res_scale)
        C = out.shape[-3]
        # res = x[...,:C,:,:] + out
        res = out
        return res
