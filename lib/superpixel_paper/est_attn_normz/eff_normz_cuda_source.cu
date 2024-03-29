#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/ATen.h>
// #include <ATen/native/cuda/KernelUtils.cuh>
// #include <ATen/native/cuda/KernelUtils.cuh>
// #include <ATen/AccumulateType.h>

#include <cuda/std/type_traits>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>


#define CUDA_NUM_THREADS 512
#define CUDA_NUM_THREADS_V2 1024
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
      acc += (samples[bi][spi][pk][si]>0) ? attn[bi][spi][hi][pi][pk] : 0;
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
    int nbatch, int nsuperpixels, int psize, int nsamples,
    int npj_per_thread, int npj_threads){

    // -- unpack --
    int bi = blockIdx.z % nbatch;
    int hi = bi / nbatch;
    int spi = blockIdx.y;
    // int ntotal = nsamples*psize*psize;//*psize;
    int ntotal = nsamples*psize*psize*npj_threads;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ntotal){ return; }
    int si = index % nsamples;
    int tmp = index / (nsamples);
    int pi = tmp % psize;
    tmp = index / (nsamples*psize);
    int pk = tmp % psize;
    // if ((si >= nsamples) or (pi >= psize) or (pk >= psize)){ return; }
    // tmp = index / (nsamples*psize*psize);
    // int pk = tmp % psize;
    // if ((si >= nsamples) or (pi >= psize) or (pj >= psize) or (pk >= psize)){ return; }
    tmp = index / (nsamples*psize*psize);
    int pj_start = npj_per_thread*(tmp % npj_threads);
    int pj_end = min(pj_start+npj_per_thread,psize);
    // if ((si >= nsamples) or (pi >= psize) or (pj >= psize) or (pk >= psize)){ return; }


    //
    // -- Compute the Estimate for (i,j) --
    //

    bool zero_grad = false;
    bool sample_is_zero = samples[bi][spi][pk][si] == 0;
    scalar_t acc = 0;
    for (int pj=pj_start;pj<pj_end;pj++){

      // -- handle indicator function for k \neq i,j --
      // bool zero_grad = false;
      // if ((pk != pi) and (pk != pj)){
      //   zero_grad = sample_is_zero;
      // }
      zero_grad = (pk != pi) and (pk != pj) and sample_is_zero;

      // -- accumulate in attention --
      acc += zero_grad ? 0 : normz_grad[bi][spi][hi][pi][pj];


    } // pj for loop

    //
    // -- Accumulate at (i,k) --
    //
    atomicAdd(&attn_grad[bi][spi][hi][pi][pk],acc/nsamples);


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

    // -- tiling --
    int npj_per_thread = psize;
    int npj_threads = (psize-1)/npj_per_thread+1;

    // -- block --
    int ntotal = nsamples*psize*psize*npj_threads;
    dim3 block((ntotal-1)/CUDA_NUM_THREADS_V2+1,nsuperpixels,nbatch*nheads);
    AT_DISPATCH_FLOATING_TYPES(attn_grad.type(), "backward_kernel", ([&] {
        eff_backward_kernel<scalar_t><<< block, CUDA_NUM_THREADS_V2 >>>(
            attn_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            normz_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            attn.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            samples.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            nbatch, nsuperpixels, psize, nsamples, npj_per_thread, npj_threads);
    }));

}


void init_eff_normz(py::module &m){
  m.def("eff_forward", &eff_forward_cuda, "efficient normz forward");
  m.def("eff_backward", &eff_backward_cuda, "efficient normz forward");
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("eff_forward", &eff_forward_cuda, "efficient normz forward");
//   m.def("eff_backward", &eff_backward_cuda, "efficient normz forward");
// }

