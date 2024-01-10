
import torch
import torch as th
import eff_normz_cuda

class EffNormzFunction(torch.autograd.Function):

    @staticmethod
    # def forward(self, attn, sims, nsamples):
    def forward(self, attn, samples):

        # -- unpack --
        B,HD,K,P0,_ = attn.shape
        attn = attn.contiguous()

        # -- allocate --
        normz = th.zeros_like(attn)

        # -- forward --
        # print(attn.shape,samples.shape,normz.shape)
        eff_normz_cuda.eff_forward(attn,samples,normz)

        # -- save --
        self.save_for_backward(samples,attn)
        return normz

    @staticmethod
    def backward(self, normz_grad):

        # -- unpack --
        samples,attn = self.saved_tensors

        # -- allocate --
        attn_grad = th.zeros_like(attn)#normz_grad)

        # -- backward --
        eff_normz_cuda.eff_backward(attn_grad,normz_grad.contiguous(),attn,samples)
        return attn_grad, None, None

