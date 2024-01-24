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

inline __host__ __device__
void check_valid(bool& valid, const int hi, const int wi, const int H, const int W){
  valid = (hi <= (H-1)) and (hi >= 0);
  valid = valid and (wi <= (W-1)) and (wi >= 0);
}

template <typename scalar_t>
__global__ void ssna_agg_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> imgOut,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> imgV,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> sims,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> sinds,
    int kernel_size){

    // -- unpack --
    int nbatch = attn.size(0);
    int nheads = attn.size(1);
    int nspix = attn.size(2);
    int height = attn.size(3);
    int width = attn.size(4);
    int ksize_sq = attn.size(4);
    int nftrs = imgV.size(4);
    int num_pix = height *width;
    int sH = sims.size(3);
    int sW = sims.size(4);

    // -- compute indices -- 
    int ibatch = blockIdx.z / nheads;
    int ihead = blockIdx.z - ibatch * nheads;
    int hw_raster = blockIdx.x * blockDim.x + threadIdx.x;
    int attn_offset = blockIdx.y * blockDim.y + threadIdx.y;
    int si = threadIdx.z;

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

    // -- read sims --
    int s_hi = sinds[hi][wi][0]+(si % 3 - 1);
    int s_wi = sinds[hi][wi][1]+(si % 3 - 1);
    bool valid = true;
    check_valid(valid,s_hi,s_wi,sH,sW);
    scalar_t sim_prob = valid ? sims[ibatch][hi][wi][s_hi][s_wi] : 0;

    // -- simplify indexing --
    auto attn_b = attn[ibatch][ihead][si][hi][wi];
    auto imgV_b = imgV[ibatch][ihead];
    scalar_t attn_val = sim_prob*attn_b[attn_offset];

    // -- accumulate --
    scalar_t val = 0;
    for(int iftr=0; iftr < nftrs; iftr++){
      val = attn_val*imgV_b[v_hi][v_wi][iftr];
      atomicAdd(&(imgOut[ibatch][ihead][hi][wi][iftr]),val);
    }
}

void ssna_agg_forward_cuda(torch::Tensor imgOut,
                            const torch::Tensor attn,
                            const torch::Tensor imgV,
                            const torch::Tensor sims,
                            const torch::Tensor sinds){

    // -- check --
    CHECK_INPUT(imgOut);
    CHECK_INPUT(attn);
    CHECK_INPUT(imgV);
    CHECK_INPUT(sims);
    CHECK_INPUT(sinds);

    // -- unpack --
    int nbatch = attn.size(0);
    int nheads = attn.size(1);
    int nspix = attn.size(2);
    int height = attn.size(3);
    int width = attn.size(4);
    int ksize_sq = attn.size(4);
    int kernel_size = std::sqrt(ksize_sq);
    int num_pix = height*width;

    // -- block --
    int nthreads_pix = 12;
    int nthreads_ksize = 8;
    int nblocks_pix = (num_pix-1)/nthreads_pix+1;
    int nblocks_ksize = (ksize_sq-1)/nthreads_ksize+1;
    dim3 nthreads(nthreads_pix,nthreads_ksize,nspix);
    dim3 nblock(nblocks_pix,nblocks_ksize,nbatch*nheads);
    AT_DISPATCH_FLOATING_TYPES(attn.type(), "forward_kernel", ([&] {
        ssna_agg_forward_kernel<scalar_t><<< nblock, nthreads >>>(
            imgOut.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            attn.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            imgV.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            sims.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            sinds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
            kernel_size);
        }));

}




template <typename scalar_t>
__global__ void ssna_agg_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> d_attn,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> d_imgV,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> d_sims,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> d_imgOut,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> imgV,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> imgOut,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> sims,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> sinds,
    int kernel_size){

    // -- unpack --
    int nbatch = attn.size(0);
    int nheads = attn.size(1);
    int height = attn.size(2);
    int width = attn.size(3);
    int ksize_sq = attn.size(4);
    int nftrs = imgV.size(4);
    int num_pix = height *width;
    int sH = sims.size(3);
    int sW = sims.size(4);


    // -- compute indices -- 
    int ibatch = blockIdx.z / nheads;
    int ihead = blockIdx.z - ibatch * nheads;
    int hw_raster = blockIdx.x * blockDim.x + threadIdx.x;
    int attn_offset = blockIdx.y * blockDim.y + threadIdx.y;
    int si = threadIdx.z;

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

    // -- read sims --
    int s_hi = sinds[hi][wi][0]+(si % 3 - 1);
    int s_wi = sinds[hi][wi][1]+(si % 3 - 1);
    bool valid = true;
    check_valid(valid,s_hi,s_wi,sH,sW);
    scalar_t sim_prob = valid ? sims[ibatch][hi][wi][s_hi][s_wi] : 0;

    // -- derivatives ("l,i" = pixel index ,"f" = feature index, "j" = attn map index) --
    scalar_t val = 0;
    scalar_t dval = 0;
    scalar_t acc_attn_grad = 0;
    scalar_t attn_val = attn[ibatch][ihead][si][hi][wi][attn_offset];
    scalar_t sim_attn_val = sim_prob*attn_val;

    for(int iftr=0; iftr < nftrs; iftr++){
      // -- [d Out[l,f]/d Vals[i,f] where i in N(l)] --
      dval = d_imgOut[ibatch][ihead][hi][wi][iftr];
      atomicAdd(&(d_imgV[ibatch][ihead][v_hi][v_wi][iftr]),dval*sim_attn_val);

      // -- [d Out[l,f]/d Attn[i,j] so i == l and sum across f] --
      val = imgV[ibatch][ihead][v_hi][v_wi][iftr];
      acc_attn_grad += val*dval;
    }
    atomicAdd(&(d_attn[ibatch][ihead][si][hi][wi][attn_offset]),acc_attn_grad);
    atomicAdd(&(d_attn[ibatch][ihead][si][hi][wi][attn_offset]),attn_val*acc_attn_grad);

}

void ssna_agg_backward_cuda(torch::Tensor d_attn,
                            torch::Tensor d_imgV,
                            torch::Tensor d_sims,
                            const torch::Tensor d_imgOut,
                            const torch::Tensor attn,
                            const torch::Tensor imgV,
                            const torch::Tensor imgOut,
                            const torch::Tensor sims,
                            const torch::Tensor sinds){

    // -- check --
    CHECK_INPUT(d_attn);
    CHECK_INPUT(d_imgV);
    CHECK_INPUT(d_sims);
    CHECK_INPUT(d_imgOut);
    CHECK_INPUT(attn);
    CHECK_INPUT(imgV);
    CHECK_INPUT(imgOut);
    CHECK_INPUT(sims);
    CHECK_INPUT(sinds);

    // -- unpack --
    int nbatch = attn.size(0);
    int nheads = attn.size(1);
    int nspix = attn.size(2);
    int height = attn.size(3);
    int width = attn.size(4);
    int ksize_sq = attn.size(4);
    int kernel_size = std::sqrt(ksize_sq);
    int num_pix = height*width;

    // -- block --
    int nthreads_pix = 12;
    int nthreads_ksize = 8;
    int nblocks_pix = (num_pix-1)/nthreads_pix+1;
    int nblocks_ksize = (ksize_sq-1)/nthreads_ksize+1;
    dim3 nthreads(nthreads_pix,nthreads_ksize,nspix);
    dim3 nblock(nblocks_pix,nblocks_ksize,nbatch*nheads);
    AT_DISPATCH_FLOATING_TYPES(attn.type(), "backward_kernel", ([&] {
        ssna_agg_backward_kernel<scalar_t><<< nblock, nthreads >>>(
            d_attn.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            d_imgV.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            d_sims.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            d_imgOut.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            attn.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            imgV.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            imgOut.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            sims.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            sinds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
            kernel_size);
        }));
}


void init_ssna_agg(py::module &m){
  m.def("ssna_agg_forward", &ssna_agg_forward_cuda,
        "neighborhood superpixel atten forward");
  m.def("ssna_agg_backward", &ssna_agg_backward_cuda,
        "neighborhood superpixel atten backward");
}
