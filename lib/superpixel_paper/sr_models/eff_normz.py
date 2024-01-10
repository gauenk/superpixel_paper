
import torch
import torch as th
# from torch.utils.cpp_extension import load_inline
# from .eff_normz_source import source
import eff_normz_cuda

# print("compile cuda source of 'pair_wise_distance' function...")
# print("NOTE: if you avoid this process, you make .cu file and compile it following https://pytorch.org/tutorials/advanced/cpp_extension.html")
# # pair_wise_distance_cuda = load_inline(
# #     "pair_wise_distance", cpp_sources="", cuda_sources=source
# # )
# print("done")


def run(attn,samples):
    B,K,P0,P1 = attn.shape
    N = samples.shape[-1]
    attn = attn.contiguous()
    samples = samples.contiguous()
    out = EffNormzFunction.apply(attn,samples)
    return out

class EffNormzFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, attn, samples):
        B,K,P0,P1 = attn.shape
        N = samples.shape[-1]
        attn = attn.contiguous()
        samples = samples.contiguous()
        normz = th.zeros((B,K,P0,P1),device=attn.device,dtype=attn.dtype)
        eff_normz_cuda.eff_forward(attn,samples,normz)
        self.save_for_backward(samples,attn)
        return normz

    @staticmethod
    def backward(self, normz_grad):
        samples,attn = self.saved_tensors
        attn_grad = th.zeros_like(normz_grad)
        eff_normz_cuda.eff_backward(attn_grad,normz_grad.contiguous(),attn,samples)
        return attn_grad, None

