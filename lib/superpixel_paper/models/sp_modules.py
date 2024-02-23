
# -- pytorch --
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_


# -- basics --
import math
from einops import rearrange

# -- superpixel --
from superpixel_paper.est_attn_normz import EffNormzFunction
from superpixel_paper.sr_models.pair_wise_distance import PairwiseDistFunction
# from spin.models.pair_wise_distance import PairwiseDistFunction
from natten import NeighborhoodAttention2D
# from .stnls_gen_sp import stnls_ssn_iter
from .ssn_model import UNet
from .lambda_model import UNet as LambdaModel
from .utils import get_pwd,calc_init_centroid
from .utils import get_hard_abs_labels,get_abs_indices


class GenSP(nn.Module):
    def __init__(self, n_iter=2,M=0.,stoken_size=8,
                 affinity_softmax=1., softmax_order="v0", use_grad=False,
                 gen_sp_type="default", dim=-1, ssn_nftrs = 9,
                 use_state=False,use_pwd=False,unet_sm=True,
                 ssn_model=None,lambda_model=None):
        super().__init__()
        self.n_iter = n_iter
        self.M = M
        self.stoken_size = stoken_size
        if isinstance(self.stoken_size,int):
            self.stoken_size = [self.stoken_size,self.stoken_size]
        self.affinity_softmax = affinity_softmax
        self.softmax_order = softmax_order
        self.use_grad = use_grad
        self.gen_sp_type = gen_sp_type
        self.use_state = use_state
        self.use_pwd = use_pwd
        self.reshape_output = gen_sp_type in ["reshape","modulated"]

        use_unet = self.gen_sp_type in ["unet","ssn"]
        use_lmodel = self.gen_sp_type in ["modulated"]
        # ssn_nftrs = 3

        # -- input dimension --
        in_dim = dim
        if use_pwd: in_dim += 9
        if use_state:
            if use_unet: in_dim += 9
            if use_lmodel: in_dim += 2

        id_l0 = nn.Identity()
        id_l1 = nn.Identity()
        if ssn_model is None:
            self.ssn = UNet(in_dim,9,ssn_nftrs,unet_sm) if use_unet else id_l0
        else:
            self.ssn = ssn_model
        if lambda_model is None:
            self.lambda_model = LambdaModel(in_dim,2,ssn_nftrs) if use_lmodel else id_l1
        else:
            self.lambda_model = lambda_model

    def forward(self, x, state=None):
        if self.gen_sp_type in ["default","reshape"]:
            _, sims, num_spixels = ssn_iter(x, self.stoken_size,
                                         self.n_iter, self.M,
                                         self.affinity_softmax,
                                         self.softmax_order,
                                         self.use_grad)
            if self.reshape_output:
                H = x.shape[-2]
                sH = H//self.stoken_size[0]
                shape_str = 'b (sh sw) (h w) -> b h w sh sw'
                sims = rearrange(sims,shape_str,h=H,sh=sH)
        elif self.gen_sp_type in ["unet","ssn"]:
            B,F,H,W = x.shape
            H = x.shape[-2]
            sH = H//self.stoken_size[0]

            if self.use_state:
                x = th.cat([x,state],-3) # stack across channels
            if self.use_pwd:
                pwd = get_pwd(x,self.stoken_size,self.affinity_softmax,self.M)
                pwd = pwd.reshape(B,9,H,W)
                # print("pwd.shape: ",pwd.shape)
                x = th.cat([x,pwd],-3) # stack across channels

            sims = self.ssn(x)
            # print(sims[0,:,64,64])
            # # print(sims.shape)
            # exit()
            if self.use_state: state = sims
            sims = sparse_to_full(sims,self.stoken_size[0])
            # print("sims.shape: ",sims.shape)
            shape_str = 'b (sh sw) (h w) -> b h w sh sw'
            sims = rearrange(sims,shape_str,h=H,sh=sH)
            num_spixels = 0 # unused

        elif self.gen_sp_type in ["modulated"]:
            B,F,H,W = x.shape


            # print(self.use_state,self.use_pwd)
            if self.use_state:
                x = th.cat([x,state],-3) # stack across channels
            if self.use_pwd:
                if self.use_state:
                    state = state.reshape(x.shape[0],2,-1)
                    M,aff = state[:,[0]],state[:,[1]]
                    M = M.reshape((B,1,H,W))
                else:
                    aff,M = self.affinity_softmax,self.M
                pwd = get_pwd(x,self.stoken_size,M,aff)
                pwd = pwd.reshape(B,9,H,W)
                # print("x.shape: ",x.shape)
                # print("pwd.shape: ",pwd.shape)
                x = th.cat([x,pwd],-3) # stack across channels

            # print("x.shape: ",x.shape)
            ssn_params = self.lambda_model(x)
            if self.use_state: state = ssn_params
            ssn_params = ssn_params.reshape(x.shape[0],2,-1)
            m_params,temp_params = ssn_params[:,[0]],ssn_params[:,[1]]
            m_params = m_params.reshape((B,1,H,W))
            # print("temp_params.shape: ",temp_params.shape)
            aff, sims, num_spixels = ssn_iter(x, self.stoken_size,
                                              self.n_iter,
                                              # self.M,
                                              m_params,
                                              temp_params,
                                              self.softmax_order,
                                              self.use_grad)
            if self.reshape_output:
                H = x.shape[-2]
                sH = H//self.stoken_size[0]
                shape_str = 'b (sh sw) (h w) -> b h w sh sw'
                sims = rearrange(sims,shape_str,h=H,sh=sH)
            # H = x.shape[-2]
            # sH = H//self.stoken_size[0]
            # sims = self.ssn(x)
            # sims = sparse_to_full(sims,self.stoken_size[0])
            # # print("sims.shape: ",sims.shape)
            # shape_str = 'b (sh sw) (h w) -> b h w sh sw'
            # sims = rearrange(sims,shape_str,h=H,sh=sH)
            # num_spixels = 0 # unused

        return sims, num_spixels, state

def get_indices(B,H,W,sH,sW,device):
    sHW = sH*sW
    labels = th.arange(sHW, device=device).reshape(1, 1, sH, sW).float()
    interp = th.nn.functional.interpolate
    labels = interp(labels, size=(H, W), mode="nearest").long()
    labels = labels.repeat(B, 1, 1, 1)
    labels = labels.reshape(B, -1)
    return labels

def sparse_to_full(sims,S):
    B,_,H,W = sims.shape
    sH,sW = H//S,W//S
    sHW = sH*sW
    ilabels = get_indices(B,H,W,sH,sW,sims.device)
    abs_indices = get_abs_indices(ilabels, sW)
    mask = (abs_indices[1] >= 0) * (abs_indices[1] < sHW)
    reshaped_affinity_matrix = sims.reshape(-1)
    sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask],
                                                  reshaped_affinity_matrix[mask])
    abs_affinity = sparse_abs_affinity.to_dense().contiguous()
    return abs_affinity


def ssn_iter(pixel_features, stoken_size=[16, 16],
             n_iter=2, M = 0., affinity_softmax=1.,
             softmax_order="v0", use_grad=False,):
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
    height, width = pixel_features.shape[-2:]
    sheight, swidth = stoken_size
    num_spixels_height = height // sheight
    num_spixels_width = width // swidth
    num_spixels = num_spixels_height * num_spixels_width

    if use_grad == "detach_x":
        pixel_features = pixel_features.detach()
    if use_grad is False: use_grad = False
    else: use_grad = True

    # -- add grid --
    from stnls.dev.slic.utils import append_grid,add_grid
    # print("pixel_features.shape: ",pixel_features.shape,M/stoken_size[0])
    # pixel_features = append_grid(pixel_features[:,None],M/stoken_size[0])[:,0]
    # print("M.shape: ",M.shape,pixel_features.shape)
    if th.is_tensor(M): M = M[:,None]
    pixel_features = append_grid(pixel_features[:,None],M/stoken_size[0])[:,0]
    # pixel_features = add_grid(pixel_features[:,None],M/stoken_size[0])[:,0]
    # print(pixel_features[:,:-2,:,:].abs().mean(),pixel_features[:,2:,:,:].abs().mean())
    # print("pixel_features.shape: ",pixel_features.shape)
    # exit()
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

    # print("use_grad: ",use_grad)
    # exit()
    with torch.set_grad_enabled(use_grad):
        for k in range(n_iter):

            # -- compute all affinities  --
            dist_matrix = PairwiseDistFunction.apply(
                    pixel_features, spixel_features, init_label_map,
                num_spixels_width, num_spixels_height)
            # print("dist_matrix.shape: ",dist_matrix.shape)
            # print(affinity_softmax.shape)
            # exit()
            if softmax_order == "v0":
                affinity_matrix = (-affinity_softmax*dist_matrix).softmax(1)
            else:
                affinity_matrix = dist_matrix

            # -- sample only relevant affinity --
            reshaped_affinity_matrix = affinity_matrix.reshape(-1)
            sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask],
                                                          reshaped_affinity_matrix[mask])
            abs_affinity = sparse_abs_affinity.to_dense().contiguous()
            if softmax_order == "v1":
                abs_affinity = (-affinity_softmax*abs_affinity).softmax(1)
            # print("abs_affinity.shape: ",abs_affinity.shape)
            # print("Ave abs affinity: ",th.mean(1.*(abs_affinity>0)).item())
            # exit()

            # -- update centroids --
            if k < n_iter - 1:
                spixel_features = torch.bmm(abs_affinity, permuted_pixel_features) \
                    / (abs_affinity.sum(2, keepdim=True) + 1e-16)

                spixel_features = spixel_features.permute(0, 2, 1).contiguous()

    return affinity_matrix, abs_affinity, num_spixels



