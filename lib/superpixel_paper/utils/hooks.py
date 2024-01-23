# -- basic --
import math
import numpy as np
import torch as th
from typing import Any, Callable



class AttentionHook():

    def __init__(self,net):
        self.net = net

        # -- known buffer names --
        self.bufs = ["q_shell","k_shell","v_shell","attn_shell",
                     "imgSp_shell_attn","imgSp_shell_agg"]
        for buf in self.bufs:
            setattr(self,buf,[])

        # -- add hook --
        for name,layer in self.net.named_modules():
            for buf in self.bufs:
                if buf in name:
                    layer.register_forward_hook(self.save_outputs_hook(buf,name))

    def save_outputs_hook(self, buffer_name: str, layer_id: str) -> Callable:
        buff = getattr(self,buffer_name)
        def fn(_, __, output):
            buff.append(output)
        return fn


def get_qkv_grads(net):
    # for name,_ in net.named_modules():
    #     print(name)
    key = "module.blocks.0.nsp_layer.1.neigh_sp_attn.nat_mat.qk"
    qk_param = [p for n,p in net.named_modules() if n == key][0]
    key = "module.blocks.0.nsp_layer.1.neigh_sp_attn.nat_agg.v"
    v_param = [p for n,p in net.named_modules() if n == key][0]
    N = qk_param.weight.grad.shape[0]
    qgrad = qk_param.weight.grad[:N//2]
    kgrad = qk_param.weight.grad[N//2:]
    vgrad = v_param.weight.grad
    # print(qgrad.shape,kgrad.shape,vgrad.shape)


    # -- viz --
    print(qgrad.abs().mean())
    print(kgrad.abs().mean())
    print(vgrad.abs().mean())

    return qgrad,kgrad,vgrad




