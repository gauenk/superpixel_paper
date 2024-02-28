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
  new_index += ((index+neigh_size)>=length) * (length - index - neigh_size - 1);
  return new_index;
}

inline __host__ __device__
void check_valid(bool& valid, const int hi, const int wi, const int H, const int W){
  valid = (hi <= (H-1)) and (hi >= 0);
  valid = valid and (wi <= (W-1)) and (wi >= 0);
}


template <typename scalar_t>
__global__ void ssna_attn_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> imgQ,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> imgK,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> sims,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> sinds,
    int kernel_size){

    // -- unpack --
    int nbatch = attn.size(0);
    int nheads = attn.size(1);
    int nsuperpixels = attn.size(2);
    int height = attn.size(3);
    int width = attn.size(4);
    int ksize_sq = attn.size(5);
    int nftrs = imgQ.size(4);
    int num_pix = height *width;
    int sH = sims.size(3);
    int sW = sims.size(4);

    // -- compute indices -- 
    int ibatch = blockIdx.z / nheads;
    int ihead = blockIdx.z - ibatch * nheads;
    int hw_raster = blockIdx.x * blockDim.x + threadIdx.x;
    int attn_offset = blockIdx.y * blockDim.y + threadIdx.y;
    int si = threadIdx.z; // ??? -- TODO --

    // -- boundary --
    if (hw_raster >= num_pix){ return; }
    if (attn_offset >= ksize_sq){ return; }

    // -- derivative indices --
    int neigh_size = (kernel_size-1)/2;
    int h_offset = attn_offset / kernel_size;
    // int w_offset = attn_offset -  h_offset * kernel_size;
    int w_offset = attn_offset % kernel_size;

    // -- compute indices --
    int hi = hw_raster / width;
    int wi = hw_raster % width;
    int v_hi = get_window_start(hi, height, neigh_size)+h_offset;
    int v_wi = get_window_start(wi, width, neigh_size)+w_offset;

    // -- read sims --
    int s_hi = sinds[hi][wi][0]+(si / 3 - 1);
    int s_wi = sinds[hi][wi][1]+(si % 3 - 1);
    bool valid = true;
    check_valid(valid,s_hi,s_wi,sH,sW);
    const scalar_t neg_inf = static_cast<scalar_t>(-INFINITY);
    scalar_t sim_prob = valid ? sims[ibatch][v_hi][v_wi][s_hi][s_wi] : neg_inf;

    // -- simplify indexing --
    auto attn_b = attn[ibatch][ihead][si][hi][wi];
    auto imgQ_pix = imgQ[ibatch][ihead][hi][wi];
    auto imgK_pix = imgK[ibatch][ihead][v_hi][v_wi];

    // -- accumulate --
    scalar_t qk_val = 0;
    for(int iftr=0; iftr < nftrs; iftr++){
      qk_val += imgQ_pix[iftr] * imgK_pix[iftr];
    }
    attn_b[attn_offset] = valid ? sim_prob*qk_val: neg_inf;
}

void ssna_attn_forward_cuda(torch::Tensor attn,
                            const torch::Tensor imgQ,
                            const torch::Tensor imgK,
                            const torch::Tensor sims,
                            const torch::Tensor sinds){

    // -- check --
    CHECK_INPUT(attn);
    CHECK_INPUT(imgQ);
    CHECK_INPUT(imgK);
    CHECK_INPUT(sims);
    CHECK_INPUT(sinds);

    // -- unpack --
    int nbatch = attn.size(0);
    int nheads = attn.size(1);
    int nsuperpixels = attn.size(2);
    assert(nsuperpixels==9);
    int height = attn.size(3);
    int width = attn.size(4);
    int ksize_sq = attn.size(5);
    int kernel_size = std::sqrt(ksize_sq);
    int num_pix = height*width;

    // -- block --
    int nthreads_pix = 14;
    int nthreads_ksize = 8;
    int nblocks_pix = (num_pix-1)/nthreads_pix+1;
    int nblocks_ksize = (ksize_sq-1)/nthreads_ksize+1;
    dim3 nthreads(nthreads_pix,nthreads_ksize,nsuperpixels);
    dim3 nblock(nblocks_pix,nblocks_ksize,nbatch*nheads);
    AT_DISPATCH_FLOATING_TYPES(attn.type(), "forward_kernel", ([&] {
        ssna_attn_forward_kernel<scalar_t><<< nblock, nthreads >>>(
            attn.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            imgQ.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            imgK.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            sims.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            sinds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
            kernel_size);
        }));
}



/***********************************************


               Backward Kernel


 ***********************************************/

template <typename scalar_t>
__global__ void ssna_attn_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> d_imgQ,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> d_imgK,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> d_sims,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> imgQ,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> imgK,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> sims,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> sinds,
    int kernel_size){

    // -- unpack --
    int nbatch = d_attn.size(0);
    int nheads = d_attn.size(1);
    int nsuperpixels = d_attn.size(2);
    int height = d_attn.size(3);
    int width = d_attn.size(4);
    int ksize_sq = d_attn.size(5);
    int nftrs = imgQ.size(4);
    int num_pix = height *width;
    int sH = sims.size(3);
    int sW = sims.size(4);

    // -- compute indices -- 
    int ibatch = blockIdx.z / nheads;
    int ihead = blockIdx.z - ibatch * nheads;
    int hw_raster = blockIdx.x * blockDim.x + threadIdx.x;
    int attn_offset = blockIdx.y * blockDim.y + threadIdx.y;
    int si = threadIdx.z; // ??? -- TODO --

    // -- boundary --
    if (hw_raster >= num_pix){ return; }
    if (attn_offset >= ksize_sq){ return; }

    // -- derivative indices --
    int neigh_size = (kernel_size-1)/2;
    int h_offset = attn_offset / kernel_size;
    // int w_offset = attn_offset -  h_offset * kernel_size;
    int w_offset = attn_offset % kernel_size;

    // -- compute indices --
    int hi = hw_raster / width;
    int wi = hw_raster % width;
    int v_hi = get_window_start(hi, height, neigh_size)+h_offset;
    int v_wi = get_window_start(wi, width, neigh_size)+w_offset;

    // -- easy access baby --
    auto imgQ_pix = imgQ[ibatch][ihead][hi][wi];
    auto imgK_pix = imgK[ibatch][ihead][v_hi][v_wi];

    // -- read sims --
    int s_hi = sinds[hi][wi][0]+(si / 3 - 1);
    int s_wi = sinds[hi][wi][1]+(si % 3 - 1);
    bool valid = true;
    check_valid(valid,s_hi,s_wi,sH,sW);
    scalar_t sim_prob = valid ? sims[ibatch][v_hi][v_wi][s_hi][s_wi] : 0;

    // -- derivatives ("l,i" = pixel index ,"f" = feature index, "j" = attn map index) --
    scalar_t acc_dQ = 0;
    scalar_t acc_dK = 0;
    scalar_t attn_val = attn[ibatch][ihead][si][hi][wi][attn_offset];
    scalar_t d_attn_val = d_attn[ibatch][ihead][si][hi][wi][attn_offset];
    scalar_t d_attn_sim_val = sim_prob*d_attn_val;
    scalar_t qk_val = 0;
    for(int iftr=0; iftr < nftrs; iftr++){
      atomicAdd(&(d_imgQ[ibatch][ihead][hi][wi][iftr]),d_attn_sim_val*imgK_pix[iftr]);
      atomicAdd(&(d_imgK[ibatch][ihead][v_hi][v_wi][iftr]),d_attn_sim_val*imgQ_pix[iftr]);
      qk_val += imgQ_pix[iftr] * imgK_pix[iftr];
    }

    // -- [dQ,dK] --
    // scalar_t eps = static_cast<scalar_t>(0.00001);
    if (valid){// and (sim_prob>0)){
      atomicAdd(&(d_sims[ibatch][v_hi][v_wi][s_hi][s_wi]),
                d_attn_val*qk_val);
      // atomicAdd(&(d_sims[ibatch][v_hi][v_wi][s_hi][s_wi]),
      //           d_attn_val*attn_val/sim_prob);
    }
}

void ssna_attn_backward_cuda(torch::Tensor d_imgQ,
                             torch::Tensor d_imgK,
                             torch::Tensor d_sims,
                             const torch::Tensor d_attn,
                             const torch::Tensor imgQ,
                             const torch::Tensor imgK,
                             const torch::Tensor attn,
                             const torch::Tensor sims,
                             const torch::Tensor sinds){

        // ctx.save_for_backward(queries, keys, attn, sims, sinds)
        // superpixel_cuda.ssna_attn_backward(
        //     d_queries,d_keys,d_sims,d_attn,
        //     ctx.saved_variables[0],
        //     ctx.saved_variables[1],
        //     ctx.saved_variables[2],
        //     ctx.saved_variables[3],
        //     ctx.saved_variables[4],
        // )

    // -- check --
    CHECK_INPUT(d_imgQ);
    CHECK_INPUT(d_imgK);
    CHECK_INPUT(d_sims);
    CHECK_INPUT(d_attn);
    CHECK_INPUT(imgQ);
    CHECK_INPUT(imgK);
    CHECK_INPUT(attn);
    CHECK_INPUT(sims);
    CHECK_INPUT(sinds);

    // -- unpack --
    int nbatch = d_attn.size(0);
    int nheads = d_attn.size(1);
    int nsuperpixels = d_attn.size(2);
    int height = d_attn.size(3);
    int width = d_attn.size(4);
    int ksize_sq = d_attn.size(5);
    int kernel_size = std::sqrt(ksize_sq);
    int num_pix = height*width;

    // -- block --
    int nthreads_pix = 12;
    int nthreads_ksize = 8;
    int nblocks_pix = (num_pix-1)/nthreads_pix+1;
    int nblocks_ksize = (ksize_sq-1)/nthreads_ksize+1;
    dim3 nthreads(nthreads_pix,nthreads_ksize,nsuperpixels);
    dim3 nblock(nblocks_pix,nblocks_ksize,nbatch*nheads);
    AT_DISPATCH_FLOATING_TYPES(d_attn.type(), "backward_kernel", ([&] {
        ssna_attn_backward_kernel<scalar_t><<< nblock, nthreads >>>(
            d_imgQ.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            d_imgK.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            d_sims.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            d_attn.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            imgQ.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            imgK.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            attn.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
            sims.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            sinds.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
            kernel_size);
        }));

}


void init_ssna_attn(py::module &m){
  m.def("ssna_attn_forward", &ssna_attn_forward_cuda,
        "neighborhood superpixel atten forward");
  m.def("ssna_attn_backward", &ssna_attn_backward_cuda,
        "neighborhood superpixel atten backward");
}
