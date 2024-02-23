
import torch
import numpy as np
import torch as th
from einops import rearrange,repeat

def append_grid(vid,R):
    B,T,F,H,W = vid.shape
    dtype,device = vid.dtype,vid.device
    grid_y, grid_x = th.meshgrid(th.arange(0, H, dtype=dtype, device=device),
                                 th.arange(0, W, dtype=dtype, device=device))
    grid = th.stack((grid_x, grid_y), -1).float()  # 2, W(x), H(y)
    grid = repeat(grid,'h w two -> b t two h w',b=B,t=T).to(device)
    vid = th.cat([vid,R*grid],2)
    return vid

def add_grid(vid,R):
    B,T,F,H,W = vid.shape
    dtype,device = vid.dtype,vid.device
    grid_y, grid_x = th.meshgrid(th.arange(0, H, dtype=dtype, device=device),
                                 th.arange(0, W, dtype=dtype, device=device))
    grid = th.stack((grid_x, grid_y), -1).float()  # 2, W(x), H(y)
    grid = rearrange(grid,'h w two -> 1 1 two h w').to(device)
    zeros = th.zeros_like(vid[:1,:1,:-2])
    # print(grid.shape,zeros.shape,vid.shape)
    to_add = th.cat([zeros,R*grid],-3)
    # vid[:,:,-2:] = vid[:,:,-2:] + R*grid
    # vid = th.cat([vid,R*grid],2)
    vid = vid + to_add
    return vid

def ndarray2tensor(ndarray_hwc):
    ndarray_chw = np.ascontiguousarray(ndarray_hwc.transpose((2, 0, 1)))
    tensor = torch.from_numpy(ndarray_chw).float()
    return tensor

