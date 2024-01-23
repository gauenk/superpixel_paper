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
__global__ void nsa_agg_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> imgOut,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> imgV,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> imgSp,
    int kernel_size){

    // -- unpack --
    int nbatch = attn.size(0);
    int nheads = attn.size(1);
    int height = attn.size(2);
    int width = attn.size(3);
    int ksize_sq = attn.size(4);
    int nftrs = imgV.size(4);
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
    // int hi = hw_raster / height;
    // int wi = hw_raster - hi * height;
    int v_hi = get_window_start(hi, height, neigh_size)+h_offset;
    int v_wi = get_window_start(wi, width, neigh_size)+w_offset;

    // -- simplify indexing --
    auto attn_b = attn[ibatch][ihead][hi][wi];
    auto imgV_b = imgV[ibatch][ihead];
    bool mask = false;
    // bool mask = (imgSp[ibatch][v_hi][v_wi] != imgSp[ibatch][hi][wi]);
    // if (mask){ return; }

    // -- accumulate --
    scalar_t val = 0;
    for(int iftr=0; iftr < nftrs; iftr++){
      val = mask ? 0 : attn_b[attn_offset]*imgV_b[v_hi][v_wi][iftr];
      atomicAdd(&(imgOut[ibatch][ihead][hi][wi][iftr]),val);
    }
}

void nsa_agg_forward_cuda(torch::Tensor imgOut,
                          const torch::Tensor attn,
                          const torch::Tensor imgV,
                          const torch::Tensor imgSp){

    // -- check --
    CHECK_INPUT(imgOut);
    CHECK_INPUT(attn);
    CHECK_INPUT(imgV);
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
        nsa_agg_forward_kernel<scalar_t><<< nblock, nthreads >>>(
            imgOut.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            attn.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            imgV.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            imgSp.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
            kernel_size);
        }));

}




template <typename scalar_t>
__global__ void nsa_agg_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> d_attn,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> d_imgV,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> d_imgSp,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> d_imgOut,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> imgV,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> imgSp,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> imgOut,
    int kernel_size){

    // -- unpack --
    int nbatch = attn.size(0);
    int nheads = attn.size(1);
    int height = attn.size(2);
    int width = attn.size(3);
    int ksize_sq = attn.size(4);
    int nftrs = imgV.size(4);
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

    // -- derivatives ("l,i" = pixel index ,"f" = feature index, "j" = attn map index) --
    scalar_t val = 0;
    scalar_t dval = 0;
    scalar_t acc_attn_grad = 0;
    scalar_t acc_imgSp_grad = 0;
    scalar_t attn_val = attn[ibatch][ihead][hi][wi][attn_offset];
    bool mask = false;
    // bool mask = (imgSp[ibatch][v_hi][v_wi] != imgSp[ibatch][hi][wi]);
    // if (mask){ return; }

    scalar_t sp_val = static_cast<scalar_t>(1);//imgSp[ibatch][ihead][v_hi][v_wi];
    for(int iftr=0; iftr < nftrs; iftr++){
      // -- [d Out[l,f]/d Vals[i,f] where i in N(l)] --
      dval = d_imgOut[ibatch][ihead][hi][wi][iftr];
      atomicAdd(&(d_imgV[ibatch][ihead][v_hi][v_wi][iftr]),dval*attn_val);

      // -- [d Out[l,f]/d Attn[i,j] so i == l and sum across f] --
      val = imgV[ibatch][ihead][v_hi][v_wi][iftr];
      acc_attn_grad += val*dval;

      // -- [d Out[l,f]/d Weight[i] so i in N(l) and sum across f] --
      acc_imgSp_grad += dval*val;

    }
    atomicAdd(&(d_attn[ibatch][ihead][hi][wi][attn_offset]),acc_attn_grad*sp_val);
    // atomicAdd(&(d_imgSp[ibatch][v_hi][v_wi]),acc_imgSp_grad*attn_val);

}

void nsa_agg_backward_cuda(torch::Tensor d_attn,
                           torch::Tensor d_imgV,
                           torch::Tensor d_imgSp,
                           const torch::Tensor d_imgOut,
                           const torch::Tensor attn,
                           const torch::Tensor imgV,
                           const torch::Tensor imgSp,
                           const torch::Tensor imgOut){

    // -- check --
    CHECK_INPUT(d_attn);
    CHECK_INPUT(d_imgV);
    CHECK_INPUT(d_imgSp);
    CHECK_INPUT(d_imgOut);
    CHECK_INPUT(attn);
    CHECK_INPUT(imgV);
    CHECK_INPUT(imgSp);
    CHECK_INPUT(imgOut);

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
    AT_DISPATCH_FLOATING_TYPES(attn.type(), "backward_kernel", ([&] {
        nsa_agg_backward_kernel<scalar_t><<< nblock, nthreads >>>(
            d_attn.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            d_imgV.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            d_imgSp.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            d_imgOut.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            attn.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            imgV.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            imgSp.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
            imgOut.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            kernel_size);
        }));
}


void init_nsa_agg(py::module &m){
  m.def("nsa_agg_forward", &nsa_agg_forward_cuda,
        "neighborhood superpixel atten forward");
  m.def("nsa_agg_backward", &nsa_agg_backward_cuda,
        "neighborhood superpixel atten backward");
}
