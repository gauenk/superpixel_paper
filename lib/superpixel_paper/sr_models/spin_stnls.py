import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dev_basics import flow
import stnls

import math
from einops import rearrange
from .pair_wise_distance import PairwiseDistFunction

from easydict import EasyDict as edict
from .share import extract
from natten import NeighborhoodAttention2D

def create_model(args):
    args = edict(vars(args))
    pairs = {"topk":None,"use_local":True,"use_intra":True,"use_inter":True,
             "use_nat":False,"nat_ksize":7,
             "softmax_order":"v0","ws":None,"base_first":False,
             "intra_version":"v1","use_ffn":True}
    extract(args,pairs)
    return SPIN(colors=args.colors, dim=args.dim, block_num=args.block_num,
                heads=args.heads, qk_dim=args.qk_dim,\
                mlp_dim=args.mlp_dim, stoken_size=args.stoken_size,
                upscale=args.upscale, M=args.M, use_local=args.use_local,
                use_inter=args.use_inter, use_intra=args.use_intra,
                use_nat=args.use_nat,nat_ksize=args.nat_ksize,
                affinity_softmax=args.affinity_softmax,topk=args.topk,ws=args.ws,
                softmax_order=args.softmax_order,base_first=args.base_first,
                intra_version=args.intra_version,use_ffn=args.use_ffn)

class SPIN(nn.Module):
    def __init__(self, colors=3, dim=40, block_num=8, heads=1, qk_dim=24, mlp_dim=72,
                 stoken_size=[12, 16, 20, 24, 12, 16, 20, 24],
                 upscale=3, M=0, use_local=True, use_inter=True, use_intra=True,
                 use_nat=False, nat_ksize=7,
                 affinity_softmax=1.,topk=None, ws=None, softmax_order="v0",
                 base_first=False,intra_version="v1",use_ffn=True):
        super(SPIN, self).__init__()
        self.dim = dim
        self.stoken_size = stoken_size
        self.block_num = block_num
        self.upscale = upscale
        if ws is None:
            ws = [None,]*len(stoken_size)
        self.base_first = base_first

        self.first_conv = nn.Conv2d(colors, dim, 3, 1, 1)

        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        for i in range(block_num):
            self.blocks.append(Block(dim=dim, layer_num=2,
                                     stoken_size=[stoken_size[i], stoken_size[i]],
                                     heads=heads, qk_dim=qk_dim, mlp_dim=mlp_dim,
                                     M=M, use_local=use_local,
                                     use_inter=use_inter, use_intra=use_intra,
                                     use_nat=use_nat,nat_ksize=nat_ksize,
                                     affinity_softmax=affinity_softmax,topk=topk,
                                     ws=ws[i],softmax_order=softmax_order,
                                     intra_version=intra_version,use_ffn=use_ffn))
            self.mid_convs.append(nn.Conv2d(dim, dim, 3, 1, 1))

        if upscale == 4:
            if base_first:
                self.upconv1 = nn.Identity()
                self.upconv2 = nn.Identity()
                self.pixel_shuffle = nn.Identity()
            else:
                self.upconv1 = nn.Conv2d(dim, dim * 4, 3, 1, 1, bias=True)
                self.upconv2 = nn.Conv2d(dim, dim * 4, 3, 1, 1, bias=True)
                self.pixel_shuffle = nn.PixelShuffle(2)
        elif upscale == 2 or upscale == 3:
            if base_first:
                self.upconv = nn.Identity()
                self.pixel_shuffle = nn.Identity()
            else:
                self.upconv = nn.Conv2d(dim, dim * (upscale ** 2), 3, 1, 1, bias=True)
                self.pixel_shuffle = nn.PixelShuffle(upscale)
        elif upscale == 1:
            self.upconv = nn.Identity()
            self.pixel_shuffle = nn.Identity()
        else:
            raise NotImplementedError(
                'Upscale factor is expected to be one of (2, 3, 4), but got {}'.format(upscale))

        self.last_conv = nn.Conv2d(dim, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        num_parameters = sum(map(lambda x: x.numel(), self.parameters()))
        print('#Params : {:<.4f} [K]'.format(num_parameters / 10 ** 3))

    def forward(self, x):
        """Forward function.
        In traning mode, 'target' should be provided for loss calculation.
        Args:
            x (Tensor): Input image.
            target (Tensor): GT image.
        """
        b, _, h, w = x.size()
        x /= 255.

        if self.upscale != 1:
            base = torch.nn.functional.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        else:
            base = x
        # print("x.shape,base.shape: ",x.shape,base.shape)
        if self.base_first:
            x = base
        x = self.first_conv(x)

        for i in range(self.block_num):
            residual = self.blocks[i](x)
            # x0 = x
            x = x + self.mid_convs[i](residual)
            # print("delta[%d]: %2.3f" % (i,(th.mean((x-x0).abs()).item())))

        if self.upscale == 4:
            out = self.pixel_shuffle(self.upconv1(x))
            out = self.pixel_shuffle(self.upconv2(out))
        else:
            out = self.pixel_shuffle(self.upconv(x))
        out = self.lrelu(out)

        out = base + self.last_conv(out)
        # print("delta[out]: %2.3f" % (th.mean((out-base).abs()).item()))
        return out * 255.


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).\
            permute(0, 3, 1, 2).contiguous()


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        dim (int): Channel number of intermediate features.
        reduction (int): Channel reduction factor. Default: 16.
    """

    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class FFN(nn.Module):
    """Feed Forward Network.
    Args:
        dim (int): Base channels.
        hidden_dim (int): Channels of hidden mlp.
    """

    def __init__(self, dim, hidden_dim, out_dim, norm_layer=LayerNorm2d):
        super().__init__()
        self.norm = norm_layer(dim)

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//8, 1, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, dim, 1, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, bias=True),
            nn.Sigmoid()
        )

        self.fc1 = nn.Conv2d(dim * 2, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = torch.cat([self.ca(x) * x, self.pa(x) * x], dim=1)
        return self.fc2(self.act(self.fc1(x)))


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

    pool_ksize = (num_spixels_height, num_spixels_width)
    # print("pool_ksize: ",pool_ksize)

    centroids = torch.nn.functional.adaptive_avg_pool2d(images, pool_ksize)
    # print("centroids.shape: ",centroids.shape)
    num_spixels = num_spixels_width * num_spixels_height

    with torch.no_grad():
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


def init_centroids_slic(vid):


    # -- init --
    vid = vid[:,None]
    stride0 = 16
    softmax_weight = 1
    use_flow = False
    wt = 0
    ws = 16
    ps = 1

    # -- init flows --
    flows = flow.orun(vid,use_flow,ftype="cv2")
    flows = stnls.nn.search_flow(flows.fflow,flows.bflow,wt,stride0)
    flows = flows[:,None].round().int()

    # -- compute pairwise searches --
    full_ws = True
    k = -1#ws*ws*(2*wt+1)
    search = stnls.search.NonLocalSearch(ws,wt,ps,k,nheads=1,dist_type="l2",
                                         stride0=stride0,#strideQ=stride0,
                                         # self_action="anchor_self",
                                         full_ws=full_ws,itype="int")
    # rvid = th.randn_like(vid)
    dists_k,flows_k = search(vid,vid,flows)
    # print(flows_k[0,0,0,0,0])
    # exit()

    # -- init slic state --
    device = vid.device
    HD = 1
    B,T,F,H,W = vid.shape
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    N = ws
    dists_k = th.ones((B,HD,T,nH,nW,N*N),device=device,dtype=th.float32)
    # flows_k = th.zeros((B,HD,T,nH,nW,N*N,3),device=device,dtype=th.int)
    # for i in range(N):
    #     for j in range(N):
    #         k = j + i*N
    #         flows_k[...,k,1] = i-(N-1)//2
    #         flows_k[...,k,2] = j-(N-1)//2
    # print(flows_k[0,0,0,0,0])
    # print(flows_k[0,0,0,0,1])

    # -- init --
    agg_ps = 1
    weights = dists_k/dists_k.sum(-1,keepdim=True)#th.softmax(-softmax_weight*dists_k,-1)
    agg = stnls.agg.NonLocalGatherAdd(agg_ps,stride0,1,
                                      outH=nH,outW=nW,itype="int")
    pooled = rearrange(agg(vid,weights,flows_k),'b hd t c h w -> b t (hd c) h w')

    return pooled

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
    # print("h,w: ",H,W)
    # nH,nW = (H-1)//region_size+1,(W-1)//region_size+1
    # import math
    # nH = int(math.sqrt(pooled.shape[-1]))
    # nW = nH
    nH,nW = (H-1)//stride0+1,(W-1)//stride0+1
    # print("pooled.shape: ",pooled.shape,nH,nW,stride0,H,W)
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

    # print(dists_k[0,0,0][8,2].shape)
    # dist_img = dists_k[0,0,0][8,2].reshape(ws,ws)
    # from dev_basics.utils import vid_io
    # vid_io.save_image(dist_img[None,:],"output/explain_rewieghting/dist_img")

    # dists_k = dists_k / F

    # search = stnls.search.NonLocalSearch(ws,wt,ps,k,nheads=1,dist_type="l2",
    #                                      stride0=stride0,strideQ=1,
    #                                      self_action="anchor_self",
    #                                      full_ws=full_ws,itype="int")
    # dists_k,flows_k = search(vid,pooled,flows)
    # print("dists_k.shape: ",dists_k.shape)


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
    # print(stoken_size)

    # print(abs_indices.shape)
    # print(abs_indices.reshape(3,16,-1).shape)
    # tmp = abs_indices.reshape(3,16,-1)
    # print(tmp.shape)
    # print(tmp[:,:,100])
    # print(abs_indices[:,:16])
    # print(abs_indices[:,16:32])
    # print(init_label_map.shape)
    # print(init_label_map.reshape(256,256)[14:20,14:20])

    # B,F = pixel_features.shape[:2]
    # stnls_features = init_centroids_slic(pixel_features).reshape(B,F,-1)
    # print(pixel_features[0,:,:16,:16].mean())
    # print(pixel_features[0,:,:16,16+1:16+16+1].mean())
    # print(pixel_features[0,:,:16,16-1:16+16-1].mean())
    # print(pixel_features[0,:,:16,16:16+16].mean())
    # print(pixel_features[0,:,16:16+16,:16].mean())
    # print("spixel_features.shape: ",spixel_features.shape)
    # print(stnls_features.shape)
    # print(spixel_features[0,:,:4])
    # print(stnls_features[0,:,:4])
    # print(spixel_features[0,:,15:18])
    # print(stnls_features[0,:,15:18])
    # # print(spixel_features[0,:,16*8+8:16*8+18])
    # # print(stnls_features[0,:,16*8+8:16*8+18])
    # exit()

    pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)
    permuted_pixel_features = pixel_features.permute(0, 2, 1).contiguous()
    # print(pixel_features.shape)
    # print(spixel_features.shape)

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

            # # # print(tmp[:,0,0])
            # # print(tmp[:,8,8])
            # # print(tmp[:,15,15])
            # # print(tmp[:,16,15])
            # # print(tmp[:,15,16])
            # # print(tmp[:,16,16])
            # # print(tmp[:,14:17,14:17])

            # # -- "scattering": expand from 9 neighbors to all centroids --
            # reshaped_affinity_matrix = affinity_matrix.reshape(-1)
            # sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask],
            #                                               reshaped_affinity_matrix[mask])
            # abs_affinity = sparse_abs_affinity.to_dense().contiguous()

            # print(abs_affinity.shape)
            # mask = abs_affinity>0
            # nz0= th.sum(stnls_affinity>0)
            # nz1= th.sum((mask*stnls_affinity)>0)
            # print(nz0,nz1)

            # # print(stnls_affinity[0,:,0])
            # # print(abs_affinity[0,:,0])
            # # print(th.sum(abs_affinity[0,:,0]>0))
            # exit()

            # abs_affinity[th.where(abs_affinity==0)] = th.inf
            # abs_affinity = (-abs_affinity).softmax(1)
            # print(abs_affinity[0,:,256*64+64])

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


class GenSP(nn.Module):
    def __init__(self, n_iter=2, M=0, ws=None, affinity_softmax=1.,softmax_order="v0"):
        super().__init__()
        self.n_iter = n_iter
        self.M = M
        self.ws = ws
        self.affinity_softmax = affinity_softmax
        self.softmax_order = softmax_order

    def forward(self, x, stoken_size):
        soft_association, num_spixels = ssn_iter(x, stoken_size,
                                                 self.n_iter, self.M,
                                                 self.affinity_softmax,self.ws,
                                                 self.softmax_order)
        return soft_association, num_spixels


class SPInterAttModule(nn.Module):
    def __init__(self, dim, num_heads, qk_dim, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.num_heads = num_heads

        self.q = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.sp = nn.Linear(dim, qk_dim, bias=qkv_bias)

        self.norm = LayerNorm2d(dim)

        head_dim = self.qk_dim // self.num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.scale = qk_scale or head_dim ** -0.5

    def forward_stoken(self, x, affinity_matrix):
        x = rearrange(x, 'b c h w -> b (h w) c')
        stokens = torch.bmm(affinity_matrix, x) / (affinity_matrix.sum(2, keepdim=True) + 1e-16) # b, n, c
        return stokens

    def forward(self, x, affinity_matrix, num_spixels):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """
        B, C, H, W = x.shape

        x = self.norm(x)

        # generate superpixel stoken
        stoken = self.forward_stoken(x, affinity_matrix) # b, k, c

        # stoken projection
        stoken = self.sp(stoken).permute(0,2,1).reshape(B, self.num_heads, self.qk_dim // self.num_heads, num_spixels) # B, H, C, hh*ww

        # q, k, v projection
        q = self.q(x).reshape(B, self.num_heads, self.qk_dim // self.num_heads, H*W) # B, H, C, H*W
        k = self.k(x).reshape(B, self.num_heads, self.qk_dim // self.num_heads, H*W) # B, H, C, H*W
        v = self.v(x).reshape(B, self.num_heads, self.dim // self.num_heads, H*W) # B, H, C, N

        # stoken interaction
        # s_attn = F.normalize(k, dim=-2).transpose(-2, -1) @ F.normalize(stoken, dim=-2) # B, H, H*W, hh*ww
        s_attn = k.transpose(-2, -1) @ stoken * self.scale # B, H, H*W, hh*ww
        s_attn = self.attn_drop(F.softmax(s_attn, -2))
        s_out = (v @ s_attn) # B, H, C, hh*ww

        # x_attn = F.normalize(stoken, dim=-2).transpose(-2, -1) @ F.normalize(q, dim=-2) # B, H, hh*ww, H*W
        x_attn = stoken.transpose(-2, -1) @ q * self.scale
        x_attn = self.attn_drop(F.softmax(x_attn, -2))
        x_out = (s_out @ x_attn).reshape(B, C, H, W) # B, H, C, H*W

        return x_out

class SPIntraAttModule(nn.Module):
    def __init__(self, dim, num_heads, qk_dim, topk=32, qkv_bias=False, qk_scale=None, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.num_heads = num_heads

        self.topk = topk

        self.q = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, 1, bias=qkv_bias)

        self.norm = LayerNorm2d(dim)

        self.attn_drop = nn.Dropout(attn_drop)
        head_dim = self.qk_dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

    def forward(self, x, affinity_matrix, num_spixels):
        B, C, H, W = x.shape

        # calculate v
        x = self.norm(x)
        q, k, v = self.q(x), self.k(x), self.v(x)

        # superpixels' pixel selection; K = # of superpixels
        # print(affinity_matrix[0][0])
        _, indices = torch.topk(affinity_matrix, self.topk, dim=-1) # B, K, topk
        # print(_misc.shape,indices.shape)
        # print(_misc[0])
        # print(_misc[0][0])
        # for i in range(3):
        #     print(_misc[0][16*7+4+i])
        #     print(indices[0][16*7+4+i])
        # for i in range(3):
        #     print(_misc[0][16*8+4+i])
        #     print(indices[0][16*8+4+i])
        # for i in range(3):
        #     print(_misc[0][20*8+4+i])
        #     print(indices[0][20*8+4+i])

        # print(self.topk)
        # print(_misc.shape)
        # print(_misc[0])
        # exit()

        q_sp_pixels = torch.gather(q.reshape(B, 1, -1, H*W).expand(-1, num_spixels, -1, -1), -1, indices.unsqueeze(2).expand(-1, -1, self.qk_dim, -1)) # B, K, C, topk
        k_sp_pixels = torch.gather(k.reshape(B, 1, -1, H*W).expand(-1, num_spixels, -1, -1), -1, indices.unsqueeze(2).expand(-1, -1, self.qk_dim, -1)) # B, K, C, topk
        v_sp_pixels = torch.gather(v.reshape(B, 1, -1, H*W).expand(-1, num_spixels, -1, -1), -1, indices.unsqueeze(2).expand(-1, -1, self.dim, -1)) # B, K, C, topk
        q_sp_pixels, k_sp_pixels, v_sp_pixels = \
            map(lambda t: rearrange(t, 'b k (h c) t -> b k h t c', h=self.num_heads),
                (q_sp_pixels, k_sp_pixels, v_sp_pixels)) # b k topk c

        # similarity
        attn = (q_sp_pixels @ k_sp_pixels.transpose(-2,-1)) * self.scale # b k h topk topk
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v_sp_pixels # b k h topk c
        out = rearrange(out, 'b k h t c -> b (h c) (k t)')
        out = scatter_mean(v.reshape(B, self.dim, H*W), -1,
                           indices.reshape(B, 1, -1).expand(-1, self.dim, -1), out)
        out = out.reshape(B, C, H, W)

        # print(out[0])
        # print(out[0].abs().mean())
        # exit()

        return out


class SPIntraAttModuleV2(nn.Module):
    def __init__(self, dim, num_heads, qk_dim, topk=32,
                 qkv_bias=False, qk_scale=None, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.num_heads = num_heads

        self.topk = topk

        self.q = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, qk_dim, 1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, 1, bias=qkv_bias)

        self.norm = LayerNorm2d(dim)

        self.attn_drop = nn.Dropout(attn_drop)
        head_dim = self.qk_dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

    def forward(self, x, affinity_matrix, num_spixels):
        B, C, H, W = x.shape

        # calculate v
        x = self.norm(x)
        q, k, v = self.q(x), self.k(x), self.v(x)

        # superpixels' pixel selection; K = # of superpixels
        # print(affinity_matrix[0][0])
        # self.topk = 300
        sims, indices = torch.topk(affinity_matrix, self.topk, dim=-1) # B, K, topk
        # print("sims.shape,indices.shape: ",sims.shape,indices.shape)
        # print("affinity_matrix.shape: ",affinity_matrix.shape)
        # sims = th.softmax(sims,-1) # B K, topk

        sample_it = lambda qkv,dim: torch.gather(qkv.reshape(B, 1, -1, H*W).expand(-1, num_spixels, -1, -1), -1, indices.unsqueeze(2).expand(-1, -1, dim, -1)) # B, K, C, topk
        shape_str = 'b k (h c) t -> b k h t c'
        reshape_it = lambda qkv: rearrange(qkv, shape_str, h=self.num_heads)
        q_sp_pixels = reshape_it(sample_it(q,self.qk_dim))
        k_sp_pixels = reshape_it(sample_it(k,self.qk_dim))
        v_sp_pixels = reshape_it(sample_it(v,self.dim))

        # -- similarity --
        attn = (q_sp_pixels @ k_sp_pixels.transpose(-2,-1)) * self.scale # b k h topk topk
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weight = sims[:,:,None,:,None].expand_as(v_sp_pixels)

        # print("sims.shape,v_sp_pixels.shape: ",sims.shape,v_sp_pixels.shape)
        # print("attn.shape,weight.shape: ",attn.shape,weight.shape)
        v_sp_pixels = weight *  v_sp_pixels
        out = attn @ v_sp_pixels # b k h topk(t) c
        # print("out.shape: ",out.shape)
        # exit()
        # weight = sims[:,:,None,:,None].expand_as(out)
        out = rearrange(out, 'b k h t c -> b (h c) (k t)')
        weight = rearrange(weight,'b k h t c -> b (h c) (k t)')
        # print(out.shape,weight.shape,H,W)
        # print(out.shape,weight.shape,v.reshape(B, self.dim, H*W).shape)
        out = weighted_scatter_mean(v.reshape(B, self.dim, H*W), weight, -1,
                                    indices.reshape(B, 1, -1).expand(-1, self.dim, -1),
                                    out)
        out = out.reshape(B, C, H, W)

        # print(out[0])
        # print(out[0].abs().mean())
        # exit()

        return out

def weighted_scatter_mean(tgt, weight, dim, indices, src):
    count = torch.ones_like(tgt)
    new_src = torch.scatter_add(tgt, dim, indices, weight*src)
    # weight = th.ones_like(weight)
    # new_count = torch.scatter_add(count, dim, indices, weight)
    # print(new_count.max())
    # new_src /= new_count
    return new_src

def scatter_mean(tgt, dim, indices, src):
    count = torch.ones_like(tgt)
    new_src = torch.scatter_add(tgt, dim, indices, src)
    new_count = torch.scatter_add(count, dim, indices, torch.ones_like(src))
    new_src /= new_count

    return new_src


class VanillaAttention(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
    """

    def __init__(self, dim, heads, qk_dim):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        """Forward function.
        If 'y' is None, it performs self-attention; Otherwise it performs cross-attention.
        Args:
            x (Tensor): Input feature.
            y (Tensor): Support feature.
        Returns:
            out(Tensor): Output feature.
        """
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out


class LocalTokenAttention(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
    """

    def __init__(self, dim, heads, qk_dim, stoken_size):
        super().__init__()

        self.stoken_size = stoken_size
        self.norm = nn.LayerNorm(dim)
        self.attn = VanillaAttention(dim, heads, qk_dim)

    def local_partition(self, x, h_step, w_step, dh, dw):
        b, c, h, w = x.size()
        local_x = []
        for i in range(0, h + h_step - dh, h_step):
            top = i
            down = i + dh
            if down > h:
                top = h - dh
                down = h
            for j in range(0, w + w_step - dw, w_step):
                left = j
                right = j + dw
                if right > w:
                    left = w - dw
                    right = w
                local_x.append(x[:, :, top:down, left:right])
        local_x = torch.stack(local_x, dim=0) 
        local_x = local_x.permute(1, 0, 2, 3, 4).contiguous()
        return local_x

    def local_reverse(self, local_x, x, h_step, w_step, dh, dw):
        b, c, h, w = x.size()
        x_output = torch.zeros_like(x)
        count = torch.zeros((b,h,w), device=x.device)

        index = 0
        for i in range(0, h + h_step - dh, h_step):
            top = i
            down = i + dh
            if down > h:
                top = h - dh
                down = h
            for j in range(0, w + w_step - dw, w_step):
                left = j
                right = j + dw
                if right > w:
                    left = w - dw
                    right = w
                x_output[:, :, top:down, left:right] += local_x[:, index]
                count[:, top:down, left:right] += 1
                index += 1
        x_output = x_output / count.unsqueeze(1)
        return x_output

    def forward(self, x):
        """Forward function.
        If 'y' is None, it performs self-attention; Otherwise it performs cross-attention.
        Args:
            x (Tensor): Input feature.
            y (Tensor): Support feature.
        Returns:
            out(Tensor): Output feature.
        """

        B, C, H, W = x.shape
        dh, dw = self.stoken_size[0], self.stoken_size[1]

        # import pdb; pdb.set_trace()
        local_x = self.local_partition(x, dh-2, dw-2, dh, dw)
        _, n, _, dh, dw = local_x.shape
        local_x = rearrange(local_x, 'b n c dh dw -> (b n) (dh dw) c')

        local_x = self.attn(self.norm(local_x))
        local_x = rearrange(local_x, '(b n) (dh dw) c  -> b n c dh dw', n=n, dw=dw)
        x = self.local_reverse(local_x, x, dh-2, dw-2, dh, dw)

        return x


class Block(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        num (int): Number of blocks.
        heads (int): Head numbers of Attention.
        qk_dim (int): Channels of query and key in Attention.
        mlp_dim (int): Channels of hidden mlp in FFN.
    """
    def __init__(self, dim, layer_num, stoken_size, heads, qk_dim, mlp_dim,
                 M=0, use_local=True, use_inter=True, use_intra=True,
                 use_nat = False, nat_ksize=7,
                 affinity_softmax=1., topk=None, ws=None, softmax_order="v0",
                 intra_version="v1",use_ffn=True):
        super(Block,self).__init__()
        self.layer_num = layer_num
        self.stoken_size = stoken_size
        self.use_inter = use_inter
        self.use_intra = use_intra
        self.use_local = use_local
        self.use_nat = use_nat
        self.use_ffn = use_ffn
        if topk is None: topk = (stoken_size[0]**2)

        self.gen_super_pixel = GenSP(3,M,ws,affinity_softmax,softmax_order)

        if self.use_inter:
            if use_ffn: ffn_l = FFN(dim, mlp_dim, dim)
            else: ffn_l = nn.Identity()
            self.inter_layer = nn.ModuleList([
                SPInterAttModule(dim, heads, qk_dim),
                ffn_l,
            ])
        else:
            self.inter_layer = nn.Identity()

        if self.use_intra:
            assert intra_version in ["v1","v2"]
            if intra_version == "v1":
                intra_l = SPIntraAttModule(dim, heads, qk_dim, topk=topk)
            elif intra_version == "v2":
                intra_l = SPIntraAttModuleV2(dim, heads, qk_dim, topk=topk)
            if use_ffn: ffn_l = FFN(dim, mlp_dim, dim)
            else: ffn_l = nn.Identity()
            self.intra_layer = nn.ModuleList([
                intra_l,ffn_l,
            ])
        else:
            self.intra_layer = nn.Identity()

        if self.use_local:
            if use_ffn: ffn_l = FFN(dim, mlp_dim, dim)
            else: ffn_l = nn.Identity()
            self.local_layer = nn.ModuleList([
                LocalTokenAttention(dim, heads, qk_dim, stoken_size),
                ffn_l,
            ])
        else:
            self.local_layer = nn.Identity()

        if self.use_nat:
            self.nat_layer = NeighborhoodAttention2D(dim=dim, kernel_size=nat_ksize,
                                                     dilation=1, num_heads=heads)
        else:
            self.nat_layer = nn.Identity()


    def forward(self, x):
        # x = self.pos_embed(x)+x

        affinity_matrix, num_spixels = self.gen_super_pixel(x, self.stoken_size)

        if self.use_inter:
            inter_attn, inter_ff = self.inter_layer
            x = inter_attn(x, affinity_matrix, num_spixels) + x
            if self.use_ffn: x = inter_ff(x) + x

        if self.use_intra:
            intra_attn, intra_ff = self.intra_layer
            # x0 = x
            x = intra_attn(x, affinity_matrix, num_spixels) + x
            if self.use_ffn: x = intra_ff(x) + x
            # print("delta: %2.3f" % (th.mean((x0-x).abs()).item()))

        if self.use_local:
            local_attn, local_ff = self.local_layer
            x = local_attn(x) + x
            if self.use_ffn: x = local_ff(x) + x

        if self.use_nat:
            from einops import rearrange
            x = rearrange(x,'b c h w -> b h w c')
            x = self.nat_layer(x)
            x = rearrange(x,'b h w c -> b c h w')

        return x
