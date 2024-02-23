"""

NLRN model for denoise dataset


"""
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import copy
dcopy = copy.deepcopy
from easydict import EasyDict as edict

from superpixel_paper.nat import NeighborhoodAttention2D
from ..ssna.menu import load_ssna
from ..models.nsp_menu import load_nsp
from ..models.ssn_model import UNet
from ..models.lambda_model import UNet as LambdaModel
from ..models.share import extract
from ..models.sp_modules import GenSP

def create_model(args):
    args = edict(vars(args))
    pairs = {"colors":3,"num_steps":12,
             "num_filters":128,"num_res_filters":64,
             "kernel_size":15,"ssn_nftrs":9,"use_proj":False,
             "gen_sp_type":"modulated","nsa_mask_labels":False,
             "topk":None,"use_local":False,"use_spa":True,
             "use_nsp":False,"use_ssna":False,
             "use_nat":False,"nat_ksize":7,"use_conv":False,
             "softmax_order":"v0","base_first":False,"spa_version":"v1",
             "use_ffn":False,"spa_scale":None,
             "spa_vweight":True,"spa_oweight":True,
             "spa_attn_nsamples":10,"use_weights":False,
             "spa_attn_normz":False,"spa_attn_normz_nsamples":10,
             "spa_scatter_normz":False,"spa_full_sampling":False,
             "use_layer_norm":True,"M":0.,"dist_type":"prod",
             "nsa_mask_labels":False,"use_midconvs":True,
             "gen_sp_use_grad":False,"gensp_niters":3,
             "use_skip":True,"gen_sp_type":"default",
             "use_attn_weights":False,"ssn_nftrs":3,"conv_ksize":3,
             "share_gen_sp":True,"heads":1,
             "use_state":False,"use_pwd":False,"use_dncnn":False,
             "unet_sm":True,"share":True,"attn_type":"default",
             "use_skip_selector":True}
    extract(args,pairs)
    model = NLRN(args,args.colors,args.num_steps,args.num_filters,
                 args.kernel_size,args.num_res_filters,args.attn_type,args.share)
    return model

class NLRN(nn.Module):

    def __init__(self, cfg, colors=3, num_steps=12,
                 num_filters=128, kernel_size=15,
                 num_res_filters=64, attn_type="default", share=False):
        super(NLRN, self).__init__()

        self.num_steps = num_steps
        self.num_filters = num_filters
        output_filter_num = num_filters

        # -- init res block --
        in_dim = num_res_filters
        gen_layers = init_gen_sp_nets(cfg.attn_type,cfg.gen_sp_type,in_dim,cfg.ssn_nftrs)
        attn_layers = init_attn_params(cfg.attn_type,num_res_filters,
                                       num_res_filters,cfg.use_proj)
        attn_block = init_attn_block(cfg,attn_type,attn_layers,gen_layers,
                                     num_res_filters,cfg.kernel_size)
        skip_sel = cfg.use_skip_selector
        res_params = init_res_params(num_res_filters,skip_sel)
        blocks = [ResidualBlock(attn_block,num_res_filters,skip_sel,*res_params)]

        for _ in range (self.num_steps-1):
            if not(share):
                # -- new parameters --
                gen_layers = init_gen_sp_nets(cfg.attn_type,cfg.gen_sp_type,
                                              num_res_filters,cfg.ssn_nftrs)
                attn_layers = init_attn_params(cfg.attn_type,num_res_filters,
                                               num_res_filters,cfg.use_proj)
                attn_block = init_attn_block(cfg,attn_type,attn_layers,gen_layers,
                                             num_res_filters,cfg.kernel_size)
                res_params = init_res_params(num_res_filters,skip_sel)
            blocks.append(ResidualBlock(attn_block,num_res_filters,skip_sel,*res_params))
        self.bn0 = torch.nn.BatchNorm2d(colors)
        self.conv_first = nn.Conv2d(colors, num_res_filters, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*blocks)
        self.bn1 = torch.nn.BatchNorm2d(num_res_filters)
        self.conv_last = nn.Conv2d(num_res_filters, colors, kernel_size=3, padding=1)

    def forward(self, x):
        skip = x
        x = self.bn0(x)
        x = self.conv_first(x)
        y = x.clone()
        for i in range(self.num_steps):
            x = self.blocks[i](x,y)
        x = F.relu(self.bn1(x))
        x = self.conv_last(x)
        return x + skip

def init_gen_sp_nets(attn_type,gen_sp_type,in_dim,ssn_nftrs):
    use_unet = gen_sp_type in ["unet","ssn"]
    use_lmodel = gen_sp_type in ["modulated"]
    if use_unet and attn_type in ["sna","ssna"]:
        ssn_model = UNet(in_dim,9,ssn_nftrs)
    else:
        ssn_model = None
    if use_lmodel and attn_type in ["sna","ssna"]:
        lambda_model = LambdaModel(in_dim,2,ssn_nftrs)
    else:
        lambda_model = None
    return ssn_model,lambda_model

def init_attn_block(cfg,attn_type,attn_layers,gen_sp_layers,
                    dim,kernel_size):
    # x_theta,x_phi,x_g,ssn_model,lambda_model):
    if attn_type == "default":
        x_theta,x_phi,x_g = attn_layers
        attn = NonLocalBlock(x_theta,x_phi,x_g,cfg.kernel_size)
    elif attn_type == "nat":
        x_theta,x_phi,x_g,proj = attn_layers
        # attn = NonLocalBlock(x_theta,x_phi,x_g,cfg.kernel_size)
        bias = False
        attn = NeighborhoodAttention2D(x_theta,x_phi,x_g,proj,
                                       dim,cfg.heads,kernel_size,
                                       dilation=1,bias=bias,qkv_bias=bias,
                                       qk_scale=cfg.spa_scale)
    elif attn_type == "sna":
        ssn_model,lambda_model = gen_sp_layers
        qk_layer,v_layer,proj_layer = attn_layers
        gen_sp = GenSP(cfg.gensp_niters,cfg.M,cfg.stoken_size,
                       cfg.affinity_softmax,cfg.softmax_order,
                       cfg.gen_sp_use_grad,cfg.gen_sp_type,
                       dim=cfg.dim,ssn_nftrs=cfg.ssn_nftrs,
                       ssn_model=ssn_model,lambda_model=lambda_model)
        attn = load_nsp(cfg.spa_version,dim,cfg.heads,dim,gen_sp,
                        topk=cfg.topk,spa_scale=cfg.spa_scale,
                        # spa_vweight=cfg.spa_vweight,
                        # spa_oweight=cfg.spa_oweight,
                        spa_attn_nsamples=cfg.spa_attn_nsamples,
                        spa_attn_normz=cfg.spa_attn_normz,
                        spa_attn_normz_nsamples=cfg.spa_attn_normz_nsamples,
                        spa_scatter_normz=cfg.spa_scatter_normz,
                        spa_full_sampling=cfg.spa_full_sampling,
                        # spa_sim_method=cfg.spa_sim_method,
                        use_proj=cfg.use_proj,
                        dist_type=cfg.dist_type,kernel_size=cfg.kernel_size,
                        mask_labels=cfg.nsa_mask_labels,use_weights=cfg.use_weights,
                        qk_layer=qk_layer,v_layer=v_layer,proj_layer=proj_layer)
    elif attn_type == "ssna":
        ssn_model,lambda_model = gen_sp_layers
        qk_layer,v_layer,proj_layer = attn_layers
        gen_sp = GenSP(cfg.gensp_niters,cfg.M,cfg.stoken_size,
                       cfg.affinity_softmax,cfg.softmax_order,
                       cfg.gen_sp_use_grad,cfg.gen_sp_type,
                       dim=cfg.dim,ssn_nftrs=cfg.ssn_nftrs,
                       ssn_model=ssn_model,lambda_model=lambda_model)
        attn = load_ssna(cfg.spa_version,dim,cfg.heads,dim,gen_sp,
                         topk=cfg.topk,spa_scale=cfg.spa_scale,
                         # spa_vweight=cfg.cfg.spa_vweight,
                         # spa_oweight=cfg.spa_oweight,
                         spa_attn_nsamples=cfg.spa_attn_nsamples,
                         spa_attn_normz=cfg.spa_attn_normz,
                         spa_attn_normz_nsamples=cfg.spa_attn_normz_nsamples,
                         spa_scatter_normz=cfg.spa_scatter_normz,
                         spa_full_sampling=cfg.spa_full_sampling,
                         # spa_sim_method=cfg.spa_sim_method,
                         use_proj=cfg.use_proj,
                         dist_type=cfg.dist_type,kernel_size=cfg.kernel_size,
                         mask_labels=cfg.nsa_mask_labels,use_weights=cfg.use_weights,
                         qk_layer=qk_layer,v_layer=v_layer,proj_layer=proj_layer)
    else:
        raise ValueError("")
    return attn

def init_attn_params(attn_type, filter_num, filter_mid_num, use_proj=False):
    if attn_type == "default":
        # print(filter_num, output_filter_num)
        x_theta = nn.Conv2d(filter_num, filter_mid_num, kernel_size=1, padding=0)
        x_phi = nn.Conv2d(filter_num, filter_mid_num, kernel_size=1, padding=0)
        x_g = nn.Conv2d(filter_num, filter_num, kernel_size=1, padding=0)
        attn_params = [x_theta,x_phi,x_g]
    elif attn_type == "nat":
        conv_q = nn.Conv2d(filter_num, filter_mid_num, kernel_size=1, padding=0)
        conv_k = nn.Conv2d(filter_num, filter_mid_num, kernel_size=1, padding=0)
        conv_v = nn.Conv2d(filter_num, filter_num, kernel_size=1, padding=0)
        if use_proj:
            proj = nn.Linear(filter_num, filter_num)
        else:
            proj = nn.Identity()
        attn_params = [conv_q,conv_k,conv_v,proj]
    elif attn_type in ["sna","ssna"]:
        qk_bias = False
        in_dim = filter_num
        qk_layer = nn.Linear(filter_num, filter_mid_num * 2, bias=qk_bias)
        v_layer = nn.Linear(filter_num, filter_num * 1, bias=qk_bias)
        if use_proj:
            proj = nn.Linear(filter_num, filter_num, bias=qk_bias)
        else:
            proj = nn.Identity()
        attn_params = [qk_layer,v_layer,proj]
    else:
        raise ValueError("")
    return attn_params

def init_res_params(num_filters,use_skip_selector):
    conv0 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
    conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
    if use_skip_selector:
        selector = SkipSelector(num_filters)
    else:
        selector = nn.Identity()
    # bn0 = torch.nn.BatchNorm2d(num_filters)
    # bn1 = torch.nn.BatchNorm2d(num_filters)
    # bn2 = torch.nn.BatchNorm2d(num_filters)
    return conv0,conv1,selector#,bn0,bn1,bn2

class ResidualBlock(th.nn.Module):

    def __init__(self,nl_block,num_filters=64,use_skip_selector=True,
                 conv0=None,conv1=None,selector=None):
        super().__init__()
        self.nl_block = nl_block

        self.use_skip_selector = use_skip_selector
        if selector:
            self.selector = selector
        elif use_skip_selector:
            self.selector = SkipSelector(num_filters)
        else:
            self.selector = nn.Identity()

        if conv0:
            self.conv0 = conv0
        else:
            self.conv0 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)

        if conv1:
            self.conv1 = conv1
        else:
            self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)

        bn0,bn1,bn2 = None,None,None
        if bn0:
            self.bn0 = bn0
        else:
            self.bn0 = torch.nn.BatchNorm2d(num_filters)

        if bn1:
            self.bn1 = bn1
        else:
            self.bn1 = torch.nn.BatchNorm2d(num_filters)

        if bn2:
            self.bn2 = bn2
        else:
            self.bn2 = torch.nn.BatchNorm2d(num_filters)

    def forward(self,x,y):
        x = self.bn0(x)
        x = F.relu(x)
        # print("[0] x.shape: ",x.shape)
        skip = x
        x = self.nl_block(x)
        if isinstance(x,tuple):
            x = x[0]
        if not(self.selector is None):
            x = self.selector(x,skip)
        else:
            x = x + skip
        # print("[1] x.shape: ",x.shape)
        x = self.bn1(x)
        x = self.conv0(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv1(x)

        x = x + y
        return x


class SkipSelector(th.nn.Module):
    """Feed Forward Network.
    Args:
        dim (int): Base channels.
        hidden_dim (int): Channels of hidden mlp.
    """

    def __init__(self, dim):
        super().__init__()
        # selector of layer
        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1, bias=True, padding=0),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, 5, bias=True, padding=2),
            nn.Sigmoid()
        )

    def forward(self,x,y):
        probs = self.selector(x)
        z = probs * x + (1 - probs)*y
        return z

class NonLocalBlock(th.nn.Module):

    def __init__(self,x_theta,x_phi,x_g,field_size):
        super().__init__()
        self.x_theta = x_theta
        self.x_phi = x_phi
        self.x_g = x_g
        self.field_size = field_size

    def forward(self, x):
        x_theta = self.x_theta(x)
        x_phi = self.x_phi(x)
        x_g = self.x_g(x)
        if self.field_size <= 0:
            assert(1 == 0)
            x_theta_reshaped = x_theta.view(x_theta.shape[0],
                                            x_theta.shape[1] * x_theta.shape[2],
                                            x_theta.shape[3])
            x_phi_reshaped = x_phi.view(x_phi.shape[0],
                                        x_phi.shape[1] * x_phi.shape[2], x_phi.shape[3])
            x_phi_permuted = torch.transpose(x_phi_reshaped, 1, 2)
            x_mul1 = torch.matmul(x_theta_reshaped, x_phi_permuted)
            x_mul1_softmax = F.softmax(x_mul1, dim=-1)
            x_g_reshaped = x_g.view(x_g.shape[0],
                                    x_g.shape[1] * x_g.shape[2], x_g.shape[3])
            x_mul2 = torch.matmul(x_mul1_softmax, x_g_reshaped)
            x_mul2 = x_mul2.view(x_mul2.shape[0], x_phi.shape[1],
                                 x_phi.shape[2], output_filter_num)
        else:
            B,_F,H,W = x_theta.shape
            x_phi_patches = F.unfold(x_phi, self.field_size, padding=self.field_size//2)
            x_phi_patches = rearrange(x_phi_patches,
                                      'b (f l) (h w) -> b h w f l',f=_F,h=H,w=W)
            x_theta = rearrange(x_theta,'b f h w -> b h w 1 f')
            x_mul1 = torch.matmul(x_theta, x_phi_patches)
            x_mul1_softmax = F.softmax(x_mul1, dim=-1)
            x_g_patches = F.unfold(x_g, self.field_size, padding=self.field_size//2)
            _F = x_g.shape[1]
            x_g_patches = rearrange(x_g_patches,'b (f l) (h w) -> b h w l f',f=_F,h=H,w=W)
            x_mul2 = torch.matmul(x_mul1_softmax, x_g_patches)
            x_mul2 = rearrange(x_mul2,'b h w 1 f -> b f h w')
        return x_mul2

