#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

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


inline __host__ __device__
int get_window_start(const int index, const int length, const int neigh_size){
  int new_index = max(index - neigh_size,0);
  new_index += (index+neigh_size>=length) * (length - index - neigh_size - 1);
  return new_index;
}

template <typename scalar_t>
__global__ void nsa_attn_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> imgQ,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> imgK,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> imgSp,
    int kernel_size){

    // -- unpack --
    int nbatch = attn.size(0);
    int nheads = attn.size(1);
    int height = attn.size(2);
    int width = attn.size(3);
    int ksize_sq = attn.size(4);
    int nftrs = imgQ.size(4);
    int num_pix = height *width;

    // -- compute indices -- 
    int ibatch = blockIdx.z / nheads;
    int ihead = blockIdx.z - ibatch * nheads;
    int hw_raster = blockIdx.x * blockDim.x + threadIdx.x;
    int attn_offset = blockIdx.y * blockDim.y + threadIdx.y;

    // -- boundary --
    if (hw_raster >= num_pix){ return; }
    if (attn_offset >= ksize_sq){ return; }

    // -- derivative indices --
    int neigh_size = (kernel_size-1)/2;
    int h_offset = attn_offset / kernel_size;
    int w_offset = attn_offset -  h_offset * kernel_size;

    // -- compute indices --
    int hi = hw_raster / width;
    int wi = hw_raster - hi * width;
    int v_hi = get_window_start(hi, height, neigh_size)+h_offset;
    int v_wi = get_window_start(wi, width, neigh_size)+w_offset;

    // -- simplify indexing --
    auto attn_b = attn[ibatch][ihead][hi][wi];
    auto imgQ_pix = imgQ[ibatch][ihead][hi][wi];
    auto imgK_pix = imgK[ibatch][ihead][v_hi][v_wi];
    bool mask = (imgSp[ibatch][v_hi][v_wi] != imgSp[ibatch][hi][wi]);
    if (mask){ return; }
    // scalar_t scale = mask ? static_cast<scalar_t>(.01) : static_cast<scalar_t>(1);

    // -- accumulate --
    scalar_t val = 0;
    for(int iftr=0; iftr < nftrs; iftr++){
      val += imgQ_pix[iftr] * imgK_pix[iftr];
    }
    attn_b[attn_offset] = val;
}

void nsa_attn_forward_cuda(torch::Tensor attn,
                           const torch::Tensor imgQ,
                           const torch::Tensor imgK,
                           const torch::Tensor imgSp){

    // -- check --
    // CHECK_INPUT(imgOut);
    CHECK_INPUT(attn);
    CHECK_INPUT(imgQ);
    CHECK_INPUT(imgK);
    CHECK_INPUT(imgSp);

    // -- unpack --
    int nbatch = attn.size(0);
    int nheads = attn.size(1);
    int height = attn.size(2);
    int width = attn.size(3);
    int ksize_sq = attn.size(4);
    int kernel_size = std::sqrt(ksize_sq);
    int num_pix = height*width;

    // -- block --
    int nthreads_pix = 96;
    int nthreads_ksize = 8;
    int nblocks_pix = (num_pix-1)/nthreads_pix+1;
    int nblocks_ksize = (ksize_sq-1)/nthreads_ksize+1;
    dim3 nthreads(nthreads_pix,nthreads_ksize);
    dim3 nblock(nblocks_pix,nblocks_ksize,nbatch*nheads);
    AT_DISPATCH_FLOATING_TYPES(attn.type(), "forward_kernel", ([&] {
        nsa_attn_forward_kernel<scalar_t><<< nblock, nthreads >>>(
            // imgOut.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            attn.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            imgQ.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            imgK.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            imgSp.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
            kernel_size);
        }));
}




template <typename scalar_t>
__global__ void nsa_attn_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> d_imgQ,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> d_imgK,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> d_imgSp,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> imgQ,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> imgK,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> imgSp,
    int kernel_size){

    // -- unpack --
    int nbatch = d_attn.size(0);
    int nheads = d_attn.size(1);
    int height = d_attn.size(2);
    int width = d_attn.size(3);
    int ksize_sq = d_attn.size(4);
    int nftrs = imgQ.size(4);
    int num_pix = height *width;

    // -- compute indices -- 
    int ibatch = blockIdx.z / nheads;
    int ihead = blockIdx.z - ibatch * nheads;
    int hw_raster = blockIdx.x * blockDim.x + threadIdx.x;
    int attn_offset = blockIdx.y * blockDim.y + threadIdx.y;

    // -- boundary --
    if (hw_raster >= num_pix){ return; }
    if (attn_offset >= ksize_sq){ return; }

    // -- derivative indices --
    int neigh_size = (kernel_size-1)/2;
    int h_offset = attn_offset / kernel_size;
    int w_offset = attn_offset -  h_offset * kernel_size;

    // -- compute indices --
    int hi = hw_raster / width;
    int wi = hw_raster - hi * width;
    int v_hi = get_window_start(hi, height, neigh_size)+h_offset;
    int v_wi = get_window_start(wi, width, neigh_size)+w_offset;

    // -- easy access baby --
    auto imgQ_pix = imgQ[ibatch][ihead][hi][wi];
    auto imgK_pix = imgK[ibatch][ihead][v_hi][v_wi];
    bool mask = (imgSp[ibatch][v_hi][v_wi] != imgSp[ibatch][hi][wi]);
    if (mask){ return; }
    // scalar_t scale = mask ? static_cast<scalar_t>(.01) : static_cast<scalar_t>(1);

    // -- derivatives ("l,i" = pixel index ,"f" = feature index, "j" = attn map index) --
    // scalar_t val = 0;
    // scalar_t dval = 0;
    // scalar_t acc_attn_grad = 0;
    // scalar_t acc_imgSp_grad = 0;
    scalar_t acc_dQ = 0;
    scalar_t acc_dK = 0;
    scalar_t d_attn_val = d_attn[ibatch][ihead][hi][wi][attn_offset];
    // scalar_t sp_val = static_cast<scalar_t>(1);//imgSp[ibatch][ihead][v_hi][v_wi];
    for(int iftr=0; iftr < nftrs; iftr++){
      atomicAdd(&(d_imgQ[ibatch][ihead][hi][wi][iftr]),d_attn_val*imgK_pix[iftr]);
      atomicAdd(&(d_imgK[ibatch][ihead][v_hi][v_wi][iftr]),d_attn_val*imgQ_pix[iftr]);
    }

    // -- [dQ,dK] --
    // atomicAdd(&(d_imgSp[ibatch][ihead][v_hi][v_wi]),acc_imgSp_grad*attn_val);
}

void nsa_attn_backward_cuda(torch::Tensor d_imgQ,
                            torch::Tensor d_imgK,
                            torch::Tensor d_imgSp,
                            const torch::Tensor d_attn,
                            const torch::Tensor imgQ,
                            const torch::Tensor imgK,
                            const torch::Tensor imgSp){

    // -- check --
    CHECK_INPUT(d_imgQ);
    CHECK_INPUT(d_imgK);
    CHECK_INPUT(d_imgSp);
    CHECK_INPUT(d_attn);
    CHECK_INPUT(imgQ);
    CHECK_INPUT(imgK);
    CHECK_INPUT(imgSp);
    // CHECK_INPUT(attn);

    // -- unpack --
    int nbatch = d_attn.size(0);
    int nheads = d_attn.size(1);
    int height = d_attn.size(2);
    int width = d_attn.size(3);
    int ksize_sq = d_attn.size(4);
    int kernel_size = std::sqrt(ksize_sq);
    int num_pix = height*width;

    // -- block --
    int nthreads_pix = 96;
    int nthreads_ksize = 8;
    int nblocks_pix = (num_pix-1)/nthreads_pix+1;
    int nblocks_ksize = (ksize_sq-1)/nthreads_ksize+1;
    dim3 nthreads(nthreads_pix,nthreads_ksize);
    dim3 nblock(nblocks_pix,nblocks_ksize,nbatch*nheads);
    AT_DISPATCH_FLOATING_TYPES(d_attn.type(), "backward_kernel", ([&] {
        nsa_attn_backward_kernel<scalar_t><<< nblock, nthreads >>>(
            d_imgQ.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            d_imgK.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            d_imgSp.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            d_attn.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            imgQ.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            imgK.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            imgSp.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
            kernel_size);
        }));

}


void init_nsa_attn(py::module &m){
  m.def("nsa_attn_forward", &nsa_attn_forward_cuda,
        "neighborhood superpixel atten forward");
  m.def("nsa_attn_backward", &nsa_attn_backward_cuda,
        "neighborhood superpixel atten backward");
}
