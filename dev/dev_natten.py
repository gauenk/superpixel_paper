
import torch as th
from natten import NeighborhoodAttention2D

dim = 9
nat_ksize = 7
heads = 1

B = 1
H,W = 32,32
img = th.randn((B,H,W,dim))

nat = NeighborhoodAttention2D(dim=dim, kernel_size=nat_ksize,
                              dilation=1, num_heads=heads)
nat(img)
