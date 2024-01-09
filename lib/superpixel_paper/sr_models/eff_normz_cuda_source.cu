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


#define CUDA_NUM_THREADS 256
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__global__ void eff_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> samples,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> out,
    int batchsize, int num_superpixels, int psize, int nsamples,
    int pk_nthreads, int num_pk){

    // -- unpack --
    int bi = blockIdx.z;
    int spi = blockIdx.y;
    int ntotal = nsamples*psize*psize*pk_nthreads;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ntotal){ return; }
    int si = index % nsamples;
    int tmp = index / (nsamples);
    int pi = tmp % psize;
    tmp = index / (nsamples*psize);
    int pj = tmp % psize;
    tmp = index / (nsamples*psize*psize);
    // int pk_init = tmp % pk_nthreads;
    int pk_init = 0;
    int PK = psize;//min(pk_init+num_pk,psize);
    if ((si >= nsamples) or (pi >= psize) or (pj >= psize)){ return; }

    scalar_t acc = 0;
    for (int pk=pk_init;pk<PK;pk++){
      // -- skip eqs --
      if (pk == pi){ continue; }
      if (pk == pj){ continue; }
      acc += attn[bi][spi][pi][pk]*samples[bi][spi][pk][si];
    }

    // -- init output --
    acc += attn[bi][spi][pi][pi];
    acc += attn[bi][spi][pi][pj];

    // -- accumulate average into (pi,pj) --
    atomicAdd(&out[bi][spi][pi][pj],acc/nsamples);

}

void eff_forward_cuda(
    const torch::Tensor attn,
    const torch::Tensor samples,
    torch::Tensor out){

    // -- check --
    CHECK_INPUT(attn);
    CHECK_INPUT(samples);
    CHECK_INPUT(out);

    // -- unpack --
    int batchsize = attn.size(0);
    int num_superpixels = attn.size(1);
    int psize = attn.size(2);
    int nsamples = samples.size(3);
    int num_pk = psize;

    // -- block --
    int pk_nthreads = (psize-1) / num_pk + 1;
    int ntotal = nsamples*psize*psize*pk_nthreads;
    dim3 block((ntotal-1)/CUDA_NUM_THREADS+1,num_superpixels,batchsize);
    AT_DISPATCH_FLOATING_TYPES(attn.type(), "forward_kernel", ([&] {
        eff_forward_kernel<scalar_t><<< block, CUDA_NUM_THREADS >>>(
            attn.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            samples.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            out.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            batchsize, num_superpixels, psize, nsamples, pk_nthreads, num_pk
        );
    }));

}



template <typename scalar_t>
__global__ void eff_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> attn_grad,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> normz_grad,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> normz,
    int batchsize, int num_superpixels, int psize){

    // -- unpack --
    int bi = blockIdx.z;
    int spi = blockIdx.y;
    int ntotal = psize*psize;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ntotal){ return; }
    int pi = index % psize;
    int tmp = index / (psize);
    int pj = tmp % psize;
    int pk_init = 0;
    int PK = psize;
    if ((pi >= psize) or (pj >= psize)){ return; }

    scalar_t acc = 0;
    for (int pk=pk_init;pk<PK;pk++){
      // -- skip eqs --
      if (pk == pi){ continue; }
      if (pk == pj){ continue; }
      acc += normz_grad[bi][spi][pi][pi]*normz[bi][spi][pi][pi];
    }

    // -- init output --
    acc += normz_grad[bi][spi][pi][pi];
    acc += normz_grad[bi][spi][pi][pj];

    // -- accumulate average into (pi,pj) --
    atomicAdd(&attn_grad[bi][spi][pi][pj],acc);

}

void eff_backward_cuda(
    torch::Tensor attn_grad,
    const torch::Tensor normz_grad,
    const torch::Tensor normz){

    // -- check --
    CHECK_INPUT(attn_grad);
    CHECK_INPUT(normz_grad);
    CHECK_INPUT(normz);

    // -- unpack --
    int batchsize = attn_grad.size(0);
    int num_superpixels = attn_grad.size(1);
    int psize = attn_grad.size(2);

    // -- block --
    // int pk_nthreads = (psize-1) / num_pk + 1;
    int ntotal = psize*psize;
    dim3 block((ntotal-1)/CUDA_NUM_THREADS+1,num_superpixels,batchsize);
    AT_DISPATCH_FLOATING_TYPES(attn_grad.type(), "backward_kernel", ([&] {
        eff_backward_kernel<scalar_t><<< block, CUDA_NUM_THREADS >>>(
            attn_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            normz_grad.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            normz.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            batchsize, num_superpixels, psize
        );
    }));

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("eff_forward", &eff_forward_cuda, "efficient normz forward");
  m.def("eff_backward", &eff_backward_cuda, "efficient normz forward");
}

