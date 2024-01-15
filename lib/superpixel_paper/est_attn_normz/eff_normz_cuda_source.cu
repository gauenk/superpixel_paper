#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda/std/type_traits>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>


#define CUDA_NUM_THREADS 512
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


template <typename scalar_t>
__global__ void eff_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> samples,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> normz,
    int nbatch, int nsuperpixels, int psize, int nsamples){

    // -- unpack --
    int bi = blockIdx.z % nbatch;
    int hi = bi / nbatch;
    int spi = blockIdx.y;
    int ntotal = nsamples*psize*psize;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ntotal){ return; }
    int si = index % nsamples;
    int tmp = index / (nsamples);
    int pi = tmp % psize;
    tmp = index / (nsamples*psize);
    int pj = tmp % psize;
    if ((si >= nsamples) or (pi >= psize) or (pj >= psize)){ return; }

    // -- compute pi --
    scalar_t acc = 0;
    for (int pk=0;pk<psize;pk++){
      // -- skip eqs --
      if (pk == pi){ continue; }
      if (pk == pj){ continue; }
      acc += attn[bi][spi][hi][pi][pk]*samples[bi][spi][pk][si];
    }

    // -- init output --
    acc += attn[bi][spi][hi][pi][pi];
    if (pi != pj){
      acc += attn[bi][spi][hi][pi][pj];
    }

    // -- accumulate average into (pi,pj) --
    // atomicAdd(&normz[bi][spi][hi][pi][pj],(1.f/acc)*(1.f/nsamples));
    atomicAdd(&normz[bi][spi][hi][pi][pj],acc*(1.f/nsamples));

}

void eff_forward_cuda(
    const torch::Tensor attn,
    const torch::Tensor samples,
    torch::Tensor normz){

    // -- check --
    CHECK_INPUT(attn);
    CHECK_INPUT(samples);
    CHECK_INPUT(normz);

    // -- unpack --
    int nbatch = attn.size(0);
    int nsuperpixels = attn.size(1);
    int nheads = attn.size(2);
    int psize = attn.size(3);
    int nsamples = samples.size(3);

    // -- block --
    int ntotal = nsamples*psize*psize;
    dim3 block((ntotal-1)/CUDA_NUM_THREADS+1,nsuperpixels,nbatch*nheads);
    AT_DISPATCH_FLOATING_TYPES(attn.type(), "forward_kernel", ([&] {
        eff_forward_kernel<scalar_t><<< block, CUDA_NUM_THREADS >>>(
            attn.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            samples.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            normz.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            nbatch, nsuperpixels, psize, nsamples
        );
    }));

}




template <typename scalar_t>
__global__ void eff_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> attn_grad,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> normz_grad,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> samples,
    int nbatch, int nsuperpixels, int psize, int nsamples){

    // -- unpack --
    int bi = blockIdx.z % nbatch;
    int hi = bi / nbatch;
    int spi = blockIdx.y;
    int ntotal = nsamples*psize*psize*psize;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ntotal){ return; }
    int si = index % nsamples;
    int tmp = index / (nsamples);
    int pi = tmp % psize;
    tmp = index / (nsamples*psize);
    int pj = tmp % psize;
    tmp = index / (nsamples*psize*psize);
    int pk = tmp % psize;
    if ((si >= nsamples) or (pi >= psize) or (pj >= psize) or (pk >= psize)){ return; }

    //
    // -- Compute the Estimate for (i,j) --
    //

    // // -- accumulate --
    // scalar_t acc = 0;
    // for (int pn=0;pn<psize;pn++){
    //   // -- skip eqs --
    //   if (pn == pi){ continue; }
    //   if (pn == pj){ continue; }
    //   // acc += attn[bi][spi][hi][pi][pn]*samples[bi][spi][pn][si];
    //   acc += samples[bi][spi][pn][si];
    // }

    // // -- doesn't use samples --
    // acc += 1;//attn[bi][spi][hi][pi][pi];
    // if (pi != pj){
    //   acc += 1;//attn[bi][spi][hi][pi][pj];
    // }

    // -- handle indicator function for k \neq i,j --
    bool zero_grad = false;
    if ((pk != pi) and (pk != pj)){
      if (samples[bi][spi][pk][si] == 0){
        zero_grad = true;
      }
    }
    scalar_t acc = zero_grad ? 0 : normz_grad[bi][spi][hi][pi][pj];
    // acc = zero_grad ? 0 : -normz_grad[bi][spi][hi][pi][pj]/(acc*acc);
    // acc = normz_grad[bi][spi][hi][pi][pj];

    //
    // -- Accumulate at (i,k) --
    //

    // -- accumulate in attention --
    atomicAdd(&attn_grad[bi][spi][hi][pi][pk],acc/nsamples);
    // atomicAdd(&attn_grad[bi][spi][hi][pi][pk],1);

}

void eff_backward_cuda(
    torch::Tensor attn_grad,
    const torch::Tensor normz_grad,
    const torch::Tensor attn,
    const torch::Tensor samples){

    // -- check --
    CHECK_INPUT(attn_grad);
    CHECK_INPUT(normz_grad);
    CHECK_INPUT(attn);
    CHECK_INPUT(samples);

    // -- unpack --
    int nbatch = attn_grad.size(0);
    int nsuperpixels = attn_grad.size(1);
    int nheads = attn_grad.size(2);
    int psize = attn_grad.size(3);
    int nsamples = samples.size(3);

    // -- block --
    int ntotal = nsamples*psize*psize*psize;
    dim3 block((ntotal-1)/CUDA_NUM_THREADS+1,nsuperpixels,nbatch*nheads);
    AT_DISPATCH_FLOATING_TYPES(attn_grad.type(), "backward_kernel", ([&] {
        eff_backward_kernel<scalar_t><<< block, CUDA_NUM_THREADS >>>(
            attn_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            normz_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            attn.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            samples.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            nbatch, nsuperpixels, psize, nsamples);
    }));

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("eff_forward", &eff_forward_cuda, "efficient normz forward");
  m.def("eff_backward", &eff_backward_cuda, "efficient normz forward");
}

