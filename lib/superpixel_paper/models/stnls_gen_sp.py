
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dev_basics import flow

import stnls

import math
from einops import rearrange
from easydict import EasyDict as edict



def calc_init_centroid(images, num_spixels_width, num_spixels_height):
    """
    calculate initial superpixels
    Args:
        images: torch.Tensor
            A Tensor of shape (B, C, H, W)
        spixels_width: int
            initial superpixel width
        spixels_height: int
            initial superpixel height
    Return:
        centroids: torch.Tensor
            A Tensor of shape (B, C, H * W)
        init_label_map: torch.Tensor
            A Tensor of shape (B, H * W)
        num_spixels_width: int
            A number of superpixels in each column
        num_spixels_height: int
            A number of superpixels int each raw
    """
    batchsize, channels, height, width = images.shape
    device = images.device

    # ones = th.ones_like(images)
    # print("ones.shape: ",ones.shape)
    centroids = torch.nn.functional.adaptive_avg_pool2d(images,\
                                    (num_spixels_height, num_spixels_width))

    with torch.no_grad():
        num_spixels = num_spixels_width * num_spixels_height
        labels = torch.arange(num_spixels, device=device).reshape(1, 1, *centroids.shape[-2:]).type_as(centroids)
        init_label_map = torch.nn.functional.interpolate(labels, size=(height, width), mode="nearest")
        init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)

    init_label_map = init_label_map.reshape(batchsize, -1)
    centroids = centroids.reshape(batchsize, channels, -1)

    return centroids, init_label_map


@torch.no_grad()
def get_abs_indices(init_label_map, num_spixels_width):
    b, n_pixel = init_label_map.shape
    device = init_label_map.device
    r = torch.arange(-1, 2.0, device=device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)

    abs_pix_indices = torch.arange(n_pixel, device=device)[None, None].repeat(b, 9, 1).reshape(-1).long()
    abs_spix_indices = (init_label_map[:, None] + relative_spix_indices[None, :, None]).reshape(-1).long()
    abs_batch_indices = torch.arange(b, device=device)[:, None, None].repeat(1, 9, n_pixel).reshape(-1).long()

    return torch.stack([abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)


@torch.no_grad()
def get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width):
    relative_label = affinity_matrix.max(1)[1]
    r = torch.arange(-1, 2.0, device=affinity_matrix.device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)
    label = init_label_map + relative_spix_indices[relative_label]
    return label.long()


def get_stnls_affinity(vid, pooled, shape, stoken_size, affinity_softmax,
                       ws, nH, nW, softmax_order):

    # -- config --
    # import math
    # H = int(math.sqrt(vid.shape[-1]))
    # nH = int(math.sqrt(pooled.shape[-1]))
    # stride0 = H//nH
    # print(nH)
    strideQ = 1
    stride0 = stoken_size[0]
    # ws = 2*stride0-1
    # ws = 2*stride0
    if ws is None:
        ws = 2*stride0-1
    # ws = 3*stride0
    ps = 1
    wt = 0
    use_flow = False
    # print("ws: ",ws)

    # -- prepare --
    vid = vid.reshape(shape)[:,None,]
    B,T,F,H,W = vid.shape
    # nH = int(math.sqrt(pooled.shape[-1]))
    # nW = nH
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    pooled = pooled.reshape((B,T,F,nH,nW))

    # -- init flows --
    flows = flow.orun(vid,use_flow,ftype="cv2")
    flows = stnls.nn.search_flow(flows.fflow,flows.bflow,wt,stride0)
    flows = flows[:,None].round().int()
    # print(flows.shape,pooled.shape,vid.shape)

    # -- compute pairwise searches --
    full_ws = False
    k = -1#ws*ws*(2*wt+1)
    search = stnls.search.NonLocalSearch(ws,wt,ps,k,nheads=1,dist_type="l2",
                                         stride0=stride0,strideQ=1,
                                         self_action="anchor_self",
                                         full_ws=full_ws,itype="int")
    dists_k,flows_k = search(pooled,vid,flows)


    # -- [transpose graph] queries to keys --
    from stnls.dev.slic import graph_transpose_q2k,graph_transpose_k2q
    outs = graph_transpose_q2k(dists_k,flows_k,flows,ws,wt,stride0,H,W,full_ws)
    scatter_dists,scatter_flows,scatter_labels = outs
    # print("scatter_dists.shape: ",scatter_dists.shape)
    # exit()

    # -- top-k --
    topk,K0 = stnls.graph_opts.scatter_topk,-1
    s_dists,s_flows,s_labels = topk(scatter_dists,scatter_flows,
                                    scatter_labels,K0,descending=False)
    # print("s_dists.shape: ",s_dists.shape)

    # s_dists = th.softmax(-affinity_softmax*s_dists,-1)
    # tmp,_ = graph_transpose_k2q(s_dists,s_flows,s_labels,stride0,T,H,W)
    # dist_img = tmp[0,0,0][8,2].reshape(ws,ws)
    # from dev_basics.utils import vid_io
    # vid_io.save_image(dist_img[None,:],"output/explain_rewieghting/dist_img_2")
    # # print("tmp.shape: ",tmp.shape)

    s_dists = s_dists.reshape((B,T,H,W,-1))
    s_dists_og = s_dists
    s_flows = s_flows.reshape((B,T,H,W,-1,3))

    if softmax_order == "v0":
        s_dists = th.softmax(-affinity_softmax*s_dists,-1)
    else:
        s_dists = s_dists

    # -- "scattering": expand from N neighbors to all centroids --
    from stnls.utils.misc import flow2inds
    # s_dists = s_dists.reshape(-1)
    # print(s_flows.shape)
    inds = th.div(flow2inds(s_flows,1)[...,1:],stride0,rounding_mode="floor")
    # print(inds[0,0,:3,:3])
    # raster = inds[...,0] + inds[...,1]*16 # wrong way
    # raster = inds[...,0]*stride0 + inds[...,1]
    # raster = inds[...,0]*nW + inds[...,1]
    raster = inds[...,0]*nW + inds[...,1]


    # print("raster.shape: ",raster.shape)
    # print(raster[th.where(th.abs(raster)<1e5)].max())
    assert raster[th.where(th.abs(raster)<1e5)].max() < (nH*nW)
    # print(raster[0,0,:3,:3].max())
    # print(raster[0,0,-3:,-3:])
    R = raster.shape[-1]
    raster = raster.view(B,-1,R)
    grid = th.arange(raster.shape[1]).view(1,-1,1).repeat(B,1,R).to(raster.device)
    # gridB = th.arange(B).view(1,1,1).repeat(1,1,R).to(raster.device)
    inds = th.stack([raster,grid],1).view(B,2,-1)
    s_dists = s_dists.view(B,-1)
    # # print(inds[0,:,:10])
    # # print(inds[0,:,-10:])
    # print(inds.shape)
    # print(s_dists.shape)
    dists = []
    # print(inds.shape)
    for bi in range(B):

        legal_i = th.abs(inds[bi,0])<1e5
        legal_i = th.logical_and(legal_i,inds[bi,0] >= 0)
        args = th.where(legal_i)[0]
        N = len(args)
        inds_bi = th.zeros((2,N)).to(inds.device)
        # print(inds_bi.shape)
        for i in range(2):
            inds_bi[i] = th.gather(inds[bi,i],0,args)
        dists_bi = th.gather(s_dists[bi],0,args)
        dists_bi = torch.sparse_coo_tensor(inds_bi,dists_bi)
        dists_bi = dists_bi.to_dense().contiguous()
        dists.append(dists_bi)
    dists = th.stack(dists)
    # print("dists.shape: ",dists.shape)

    # -- check --
    # check = th.sort(dists,0,descending=True).values[:4]
    # check = check.T.reshape(H,W,4)
    # print(check.shape)
    # s_dists = s_dists_og.reshape((B,T,H,W,-1))[0,0]
    # print("dists.shape: ",dists.shape)
    # dist_img = dists[0].reshape(nH,nW,H,W)[8,2]
    # vid_io.save_image(dist_img[None,:],"output/explain_rewieghting/dist_img_f")

    # print(check[:3,:3])
    # print(s_dists[:3,:3])
    # # exit()

    return dists

    # return dists_k.view(B,-1,ws*ws).transpose(1,2)

def ssn_iter(pixel_features, stoken_size=[16, 16], n_iter=2, M=0.,
             affinity_softmax=1., ws=None, softmax_order="v0"):
    """
    computing assignment iterations
    detailed process is in Algorithm 1, line 2 - 6
    Args:
        pixel_features: torch.Tensor
            A Tensor of shape (B, C, H, W)
        num_spixels: int
            A number of superpixels
        n_iter: int
            A number of iterations
        return_hard_label: bool
            return hard assignment or not
    """

    # -- unpack --
    height, width = pixel_features.shape[-2:]
    sheight, swidth = stoken_size
    # num_spixels_height = height // sheight
    # num_spixels_width = width // swidth
    num_spixels_height = (height-1) // sheight+1
    num_spixels_width = (width-1) // swidth+1
    # print(height,sheight,swidth,num_spixels_height,num_spixels_width)
    num_spixels = num_spixels_height * num_spixels_width
    nH,nW = num_spixels_height,num_spixels_width
    # print("num_spixels_height,num_spixels_width: ",
    #       num_spixels_height,num_spixels_width)

    # -- add grid --
    from stnls.dev.slic.utils import append_grid
    # print("pixel_features.shape: ",pixel_features.shape,M/stoken_size[0])
    pixel_features = append_grid(pixel_features[:,None],M/stoken_size[0])[:,0]
    # print("pixel_features.shape: ",pixel_features.shape)
    shape = pixel_features.shape
    # th.cuda.synchronize()
    assert softmax_order in ["v0","v1"],"softmax order must be v0,v1"

    # -- init centroids/inds --
    spixel_features, init_label_map = \
        calc_init_centroid(pixel_features, num_spixels_width, num_spixels_height)
    abs_indices = get_abs_indices(init_label_map, num_spixels_width)
    mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels)

    # -- formating features --
    pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)
    permuted_pixel_features = pixel_features.permute(0, 2, 1).contiguous()

    # -- not grad --
    with torch.no_grad():
        for k in range(n_iter):

            # # -- compute affinities with 9 neighboring centroids  --
            # dist_matrix = PairwiseDistFunction.apply(
            #     pixel_features, spixel_features, init_label_map,
            #     num_spixels_width, num_spixels_height)
            # affinity_matrix = (-dist_matrix).softmax(1)
            # # print("affinity_matrix.shape: ",affinity_matrix.shape)
            # # affinity_matrix = dist_matrix
            # # tmp = affinity_matrix.reshape(9,256,256)
            # # print(affinity_matrix.shape)
            # # print(tmp.shape)

            # -- stnls affinity --
            stnls_affinity = get_stnls_affinity(pixel_features, spixel_features,
                                                shape, stoken_size,
                                                affinity_softmax, ws, nH, nW,
                                                softmax_order)
            if softmax_order == "v0":
                abs_affinity = stnls_affinity
            elif softmax_order == "v1":
                abs_affinity = (-affinity_softmax*stnls_affinity).softmax(1)
            else:
                raise ValueError(f"Uknown softmax order [{softmax_order}]")
            # abs_affinity = (-affinity_softmax*stnls_affinity).softmax(1)
            # print("Ave abs affinity: ",th.mean(1.*(abs_affinity>0)).item())
            # exit()
            # tmp = stnls_affinity.reshape(9,256,256)
            # print(stnls_affinity.shape)
            # print(tmp[:,0,0])
            # print(tmp[:,8,8])

            # -- update centroids --
            if k < n_iter - 1:
                spixel_features = torch.bmm(abs_affinity, permuted_pixel_features) \
                    / (abs_affinity.sum(2, keepdim=True) + 1e-16)

                spixel_features = spixel_features.permute(0, 2, 1).contiguous()
            # print("spixel_features.shape: ",spixel_features.shape)

    # print("abs_affinity.shape: ",abs_affinity.shape)
    return abs_affinity, num_spixels


