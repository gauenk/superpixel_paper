
# -- import basic --
import torch as th
import numpy as np

# -- import stnls --
import stnls

# -- compare sims --
from superpixel_modes.sr_models.spin import ssn_iter(pixel_features, stoken_size=[16, 16], n_iter=2, M = 0.,
             affinity_softmax=1., softmax_order="v0",use_grad=False)


def ssn_affinity(pixel_features,spixel_features,init_label_map,
                 num_spixels_width,num_spixels_height,sm_scale):
    # -- compute all affinities  --
    dist_matrix = PairwiseDistFunction.apply(
            pixel_features, spixel_features, init_label_map,
        num_spixels_width, num_spixels_height)
    affinity_matrix = (-sm_scale*dist_matrix).softmax(1)

    # -- sample only relevant affinity --
    reshaped_affinity_matrix = affinity_matrix.reshape(-1)
    sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask],
                                                  reshaped_affinity_matrix[mask])
    abs_affinity = sparse_abs_affinity.to_dense().contiguous()
    return abs_affinity

def stnls_affinity(vid, pooled, shape, stoken_size, sm_scale, ws):

    # -- config --
    strideQ = 1
    stride0 = stoken_size[0]
    if ws is None: ws = 2*stride0-1
    ps,wt,use_flow = 1,0,False

    # -- prepare --
    vid = vid.reshape(shape)[:,None,]
    B,T,F,H,W = vid.shape
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    pooled = pooled.reshape((B,T,F,nH,nW))

    # -- init flows --
    flows = flow.orun(vid,use_flow,ftype="cv2")
    flows = stnls.nn.search_flow(flows.fflow,flows.bflow,wt,stride0)
    flows = flows[:,None].round().int()

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

    # -- top-k --
    topk,K0 = stnls.graph_opts.scatter_topk,-1
    s_dists,s_flows,s_labels = topk(scatter_dists,scatter_flows,
                                    scatter_labels,K0,descending=False)
    s_dists = s_dists.reshape((B,T,H,W,-1))
    s_dists = th.softmax(-sm_scale*s_dists,-1)
    s_flows = s_flows.reshape((B,T,H,W,-1,3))


def sparse2full_stnls(s_dists,s_flows,stride0,nH,nW):

    # -- "scattering": expand from N neighbors to all centroids --
    from stnls.utils.misc import flow2inds
    inds = th.div(flow2inds(s_flows,1)[...,1:],stride0,rounding_mode="floor")
    raster = inds[...,0]*nW + inds[...,1]
    assert raster[th.where(th.abs(raster)<1e5)].max() < (nH*nW)
    R = raster.shape[-1]
    raster = raster.view(B,-1,R)
    grid = th.arange(raster.shape[1]).view(1,-1,1).repeat(B,1,R).to(raster.device)
    inds = th.stack([raster,grid],1).view(B,2,-1)
    s_dists = s_dists.view(B,-1)
    dists = []
    for bi in range(B):
        legal_i = th.abs(inds[bi,0])<1e5
        legal_i = th.logical_and(legal_i,inds[bi,0] >= 0)
        args = th.where(legal_i)[0]
        N = len(args)
        inds_bi = th.zeros((2,N)).to(inds.device)
        for i in range(2):
            inds_bi[i] = th.gather(inds[bi,i],0,args)
        dists_bi = th.gather(s_dists[bi],0,args)
        dists_bi = torch.sparse_coo_tensor(inds_bi,dists_bi)
        dists_bi = dists_bi.to_dense().contiguous()
        dists.append(dists_bi)
    dists = th.stack(dists)
    return dists


def ssn_iter_setup(pixel_features,stoken_size):
    height, width = pixel_features.shape[-2:]
    sheight, swidth = stoken_size
    num_spixels_height = height // sheight
    num_spixels_width = width // swidth
    num_spixels = num_spixels_height * num_spixels_width

    # -- add grid --
    from stnls.dev.slic.utils import append_grid
    # print("pixel_features.shape: ",pixel_features.shape,M/stoken_size[0])
    pixel_features = append_grid(pixel_features[:,None],M/stoken_size[0])[:,0]
    # print("pixel_features.shape: ",pixel_features.shape)
    shape = pixel_features.shape

    # import pdb; pdb.set_trace()
    # num_spixels_width = int(math.sqrt(num_spixels * width / height))
    # num_spixels_height = int(math.sqrt(num_spixels * height / width))

    # -- init centroids/inds --
    spixel_features, init_label_map = \
        calc_init_centroid(pixel_features, num_spixels_width, num_spixels_height)
    abs_indices = get_abs_indices(init_label_map, num_spixels_width)
    mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels)

    pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)
    permuted_pixel_features = pixel_features.permute(0, 2, 1).contiguous()
    assert softmax_order in ["v0","v1"], "Softmax order must be v0, v1."
    return pixel_features,spixel_features,num_spixels_height,\
        num_spixels_width,init_label_map

def main():


    # -- setup --
    shape = img.shape
    pixel_features = img
    stoken_size = [8,8]
    sm_scale = 1
    stride0 = stoken_size[0]
    ws = 24

    # -- compare cluster sims --
    pix,spix,sH,sW,ilabel_map = ssn_iter_setup(pixel_features,stoken_size,sm_scale)
    sims_ssn = ssn_affinity(pix,spix,ilabel_map,sH,sW)
    sims_stnls_s,s_flows = stnls_affinity(pix,spix,shape,stride0,sm_scale,ws)
    sims_stnls = sparse2full_stnls(sims_stnls_s,s_flows,stride0,nH,nW)
    sims_stnls = sparse2full_sims(sims_stnls_sparse)
    error = th.mean((sims_ssn - sims_stnls)**2)
    print(error)


if __name__ == "__main__":
    main()
