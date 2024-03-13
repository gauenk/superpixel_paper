# -- import torch --
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# -- basic --
import math
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# -- types --
from typing import Any, Callable
from easydict import EasyDict as edict


class SsnaSuperpixelHooks():

    def __init__(self,net):
        self.net = net
        self.attn = net.blocks[0].ssna_layer[1].neigh_sp_attn
        self.spix = [None]

        # -- register hooks with buffer names --
        layer = net.blocks[0].ssna_layer[1].neigh_sp_attn
        layer.register_forward_hook(self.save_outputs_hook(self.spix))

    def save_outputs_hook(self, buff) -> Callable:
        def fn(_, inputs, output):
            buff[0] = inputs[1]
        return fn


