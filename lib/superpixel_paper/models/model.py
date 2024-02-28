# -- import torch --
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# -- basic --
import math
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# from natten import NeighborhoodAttention2D
from superpixel_paper.nat.nat_spin import NeighborhoodAttention2D
# from ..nat import NeighborhoodAttention2D
from positional_encodings.torch_encodings import PositionalEncoding2D
from positional_encodings.torch_encodings import PositionalEncodingPermute2D


# -- submodules --
from .share import extract
from .spa_menu import load_spa
from ..sna.menu import load_sna
from ..ssna.menu import load_ssna
from .sp_modules import GenSP
from .bass import SimulateBass
from .guts import LocalTokenAttention,FFN,LayerNorm2d
from .dncnn import DnCNN
from .res import ResBlockList
from .lambda_model import UNet as AttentionScaleNet


def create_model(args):
    args = edict(vars(args))
    pairs = {"topk":None,"use_local":False,"use_spa":True,
             "use_sna":False,
             "use_nsp":False,"use_ssna":False,
             "use_nat":False,"nat_ksize":7,"use_conv":False,
             "softmax_order":"v0","base_first":False,"spa_version":"v1",
             "use_ffn":False,"spa_scale":None,
             "spa_vweight":True,"spa_oweight":True,
             "spa_attn_nsamples":10,
             "spa_attn_normz":False,"spa_attn_normz_nsamples":10,
             "spa_scatter_normz":False,"spa_full_sampling":False,
             "use_layer_norm":True,"M":0.,"dist_type":"prod",
             "nsa_mask_labels":False,"use_midconvs":True,
             "gen_sp_use_grad":False,"gensp_niters":3,
             "use_skip":True,"gen_sp_type":"default",
             "use_attn_weights":False,"ssn_nftrs":3,"conv_ksize":3,
             "share_gen_sp":True,"heads":1,
             "use_state":False,"use_pwd":False,"use_dncnn":False,
             "unet_sm":True,"use_proj":True,"learn_attn_scale":False,
             "detach_sims":False,"detach_learn_attn":False}
    extract(args,pairs)
    # print({k:args[k] for k in pairs})
    return SimpleModel(colors=args.colors, dim=args.dim, block_num=args.block_num,
                       heads=args.heads, qk_dim=args.qk_dim,
                       mlp_dim=args.mlp_dim, stoken_size=args.stoken_size,
                       upscale=args.upscale, M=args.M, use_local=args.use_local,
                       use_spa=args.use_spa,use_ssna=args.use_ssna,
                       use_nat=args.use_nat,nat_ksize=args.nat_ksize,
                       use_conv=args.use_conv,
                       affinity_softmax=args.affinity_softmax,topk=args.topk,
                       softmax_order=args.softmax_order,base_first=args.base_first,
                       spa_version=args.spa_version,use_ffn=args.use_ffn,
                       spa_vweight=args.spa_vweight,
                       spa_oweight=args.spa_oweight,spa_scale=args.spa_scale,
                       spa_attn_nsamples=args.spa_attn_nsamples,
                       spa_attn_normz=args.spa_attn_normz,
                       spa_attn_normz_nsamples=args.spa_attn_normz_nsamples,
                       spa_scatter_normz=args.spa_scatter_normz,
                       spa_full_sampling=args.spa_full_sampling,
                       use_layer_norm=args.use_layer_norm,
                       dist_type=args.dist_type,use_nsp=args.use_nsp,
                       nsa_mask_labels=args.nsa_mask_labels,
                       use_midconvs=args.use_midconvs,
                       gen_sp_use_grad=args.gen_sp_use_grad,
                       gensp_niters=args.gensp_niters,
                       use_skip=args.use_skip,
                       gen_sp_type=args.gen_sp_type,
                       use_weights=args.use_attn_weights,
                       ssn_nftrs=args.ssn_nftrs,conv_ksize=args.conv_ksize,
                       share_gen_sp=args.share_gen_sp,
                       use_state=args.use_state,use_pwd=args.use_pwd,
                       use_dncnn=args.use_dncnn,unet_sm=args.unet_sm,
                       use_proj=args.use_proj,use_sna=args.use_sna,
                       learn_attn_scale=args.learn_attn_scale,
                       detach_sims=args.detach_sims,
                       detach_learn_attn=args.detach_learn_attn)

class SimpleModel(nn.Module):
    def __init__(self, colors=3, dim=40, block_num=8, heads=1, qk_dim=24, mlp_dim=72,
                 stoken_size=[12, 16, 20, 24, 12, 16, 20, 24], upscale=3,
                 M=0., use_local=True, use_spa=True, use_ssna=False,
                 use_nat=False, nat_ksize=7, use_conv=False,
                 affinity_softmax=1.,topk=None,softmax_order="v0",
                 base_first=False,spa_version="v1",use_ffn=True,
                 spa_vweight=True,spa_oweight=True,spa_scale=None,
                 spa_attn_nsamples=10,
                 spa_attn_normz=None,spa_attn_normz_nsamples=10,
                 spa_normz_nsamples=10,spa_normz_version="map",
                 spa_scatter_normz=None,spa_full_sampling=False,
                 use_layer_norm=True,dist_type="l2",use_nsp=False,
                 nsa_mask_labels=False,use_midconvs=True,
                 gen_sp_use_grad=False,gensp_niters=3,
                 use_skip=True,gen_sp_type="default",
                 conv_ksize=3,use_weights=False,ssn_nftrs=3,
                 share_gen_sp=True,use_state=False,
                 use_pwd=False,use_dncnn=True,unet_sm=True,
                 use_proj=True,use_sna=False,learn_attn_scale=False,
                 detach_sims=False,detach_learn_attn=False):
        super(SimpleModel, self).__init__()

        # -- simplify stoken_size specification --
        if isinstance(stoken_size,int):
            stoken_size = [stoken_size,]*block_num
        if (len(stoken_size) == 1):
            stoken_size = stoken_size*block_num
        assert len(stoken_size) == block_num

        self.dim = dim
        self.stoken_size = stoken_size
        self.block_num = block_num
        self.upscale = upscale
        self.base_first = base_first
        self.use_midconvs = use_midconvs
        conv_pad = conv_ksize//2
        self.first_conv = nn.Conv2d(colors, dim, conv_ksize, 1, conv_pad)

        # -- init state for gen sp --
        self.gen_sp_type = gen_sp_type
        self.use_state = use_state
        self.affinity_softmax = affinity_softmax
        self.M = M

        # self.first_conv = nn.Conv2d(colors, dim, 1, 1, 0)
        # self.first_conv = nn.Identity()
        # self.res_block = ResBlockList(3, dim, 3, bn=False)
        self.use_dncnn = use_dncnn
        self.dncnn = DnCNN(colors) if use_dncnn else nn.Identity()
        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        self.pre_conv = nn.Identity()
        self.post_conv = nn.Identity()
        self.residual_shell = nn.Identity()
        self.lrelu_shell = nn.Identity()
        self.last_conv_shell = nn.Identity()


        # share_gen_sp = True
        self.share_gen_sp = share_gen_sp
        if self.share_gen_sp:
            # print(stoken_size)
            # print(np.all(np.array(stoken_size)==stoken_size[0]))
            assert np.all(np.array(stoken_size)==stoken_size[0])
            _ssize = [stoken_size[0],stoken_size[0]]
            self.gen_sp = GenSP(gensp_niters,M,_ssize,affinity_softmax,
                                softmax_order,gen_sp_use_grad,gen_sp_type,
                                dim=dim,ssn_nftrs=ssn_nftrs,
                                use_state=use_state,use_pwd=use_pwd)
        else:
            self.gen_sp = None

        for i in range(block_num):
            self.blocks.append(Block(dim=dim, layer_num=2,
                                     stoken_size=[stoken_size[i], stoken_size[i]],
                                     heads=heads, qk_dim=qk_dim, mlp_dim=mlp_dim,
                                     M=M, use_local=use_local,
                                     use_spa=use_spa,use_ssna=use_ssna,
                                     use_nat=use_nat,nat_ksize=nat_ksize,
                                     use_conv=use_conv,
                                     affinity_softmax=affinity_softmax,topk=topk,
                                     softmax_order=softmax_order,
                                     spa_version=spa_version,use_ffn=use_ffn,
                                     spa_vweight=spa_vweight,spa_oweight=spa_oweight,
                                     spa_scale=spa_scale,
                                     spa_attn_nsamples=spa_attn_nsamples,
                                     spa_attn_normz=spa_attn_normz,
                                     spa_attn_normz_nsamples=spa_attn_normz_nsamples,
                                     spa_scatter_normz=spa_scatter_normz,
                                     spa_full_sampling=spa_full_sampling,
                                     use_layer_norm=use_layer_norm,
                                     dist_type=dist_type,use_nsp=use_nsp,
                                     use_sna=use_sna,
                                     nsa_mask_labels=nsa_mask_labels,
                                     gen_sp_use_grad=gen_sp_use_grad,
                                     gensp_niters=gensp_niters,
                                     use_skip=use_skip,gen_sp_type=gen_sp_type,
                                     use_weights=use_weights,ssn_nftrs=ssn_nftrs,
                                     share_gen_sp=self.gen_sp,unet_sm=unet_sm,
                                     use_proj=use_proj,learn_attn_scale=learn_attn_scale,
                                     detach_sims=detach_sims,
                                     detach_learn_attn=detach_learn_attn))
            if self.use_midconvs:
                conv_pad = conv_ksize//2
                self.mid_convs.append(nn.Conv2d(dim, dim, conv_ksize, 1, conv_pad))
            else:
                self.mid_convs.append(nn.Identity())

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
            msg = 'Upscale factor is expected to be one'
            msg += '  of (2, 3, 4), but got {}'.format(upscale)
            raise NotImplementedError(msg)

        if not(self.use_dncnn):
            conv_pad = conv_ksize//2
            self.last_conv = nn.Conv2d(dim, 3, conv_ksize, 1, conv_pad)
            # self.last_conv = nn.Conv2d(dim, 3, 1, 1, 0)
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            self.last_conv = nn.Identity()
            self.lrelu = nn.Identity()
        num_parameters = sum(map(lambda x: x.numel(), self.parameters()))
        print('#Params : {:<.4f} [K]'.format(num_parameters / 10 ** 3))

    # def freeze_scale_net(self):
    #     for block in self.blocks[i].named_Child:

    #     pass

    def forward(self, x):
        """Forward function.
        In traning mode, 'target' should be provided for loss calculation.
        Args:
            x (Tensor): Input image.
            target (Tensor): GT image.
        """
        b, _, h, w = x.size()
        x = x / 255.
        if self.use_dncnn:
            return self.dncnn(x)*255.

        if self.upscale != 1:
            base = torch.nn.functional.interpolate(x, scale_factor=self.upscale,
                                                   mode='bilinear', align_corners=False)
        else:
            base = x
        # print("base.shape: ",base.shape,x.shape)
        # print("."*30)

        if self.base_first: x = base
        x = self.pre_conv(x)
        x = self.first_conv(x)
        x = self.post_conv(x)
        # x = self.res_block(x)

        state = self.init_state(x)
        for i in range(self.block_num):
            residual,state = self.blocks[i](x,state)
            if self.use_midconvs:
                x = x + self.mid_convs[i](residual)
            else:
                x = x + self.residual_shell(residual)
        # print(x.shape)
        # exit()

        if self.upscale == 4:
            out = self.pixel_shuffle(self.upconv1(x))
            out = self.pixel_shuffle(self.upconv2(out))
        else:
            out = self.pixel_shuffle(self.upconv(x))
        out = self.lrelu_shell(self.lrelu(out))
        out = base + self.last_conv_shell(self.last_conv(out))
        # out = self.last_conv(out)

        return out * 255.

    def init_state(self,x):
        use_unet = self.gen_sp_type in ["unet","ssn"]
        use_lmodel = self.gen_sp_type in ["modulated"]
        aff = self.affinity_softmax
        M = self.M
        state = None
        B,_,H,W = x.shape
        if self.use_state:
            if use_unet:
                state = th.ones((B,9,H,W),device=x.device,dtype=x.dtype)/9.
            elif use_lmodel:
                state = th.ones((B,2,H,W),device=x.device,dtype=x.dtype)
                state[:,0] = M*state[:,0]
                state[:,1] = aff*state[:,1]
        return state

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
                 M=0., use_local=False, use_spa=True, use_ssna=False,
                 use_nat=False, nat_ksize=7, use_conv=False,
                 affinity_softmax=1., topk=None, softmax_order="v0",
                 spa_version="v1",use_ffn=True,spa_scale=None,
                 spa_vweight=True,spa_oweight=True,
                 spa_attn_nsamples=10,spa_attn_normz=None,
                 spa_attn_normz_nsamples=10,spa_scatter_normz=None,
                 spa_full_sampling=False,spa_sim_method="slic",
                 use_layer_norm=True,dist_type="l2",use_nsp=False,
                 use_sna=False,
                 nsa_mask_labels=False,gen_sp_use_grad=False,
                 gensp_niters=3,use_skip=True,gen_sp_type="default",
                 use_weights=True,ssn_nftrs=3,share_gen_sp=None,
                 unet_sm=True,use_proj=True,learn_attn_scale=False,
                 detach_sims=False,detach_learn_attn=False):
        super(Block,self).__init__()
        self.layer_num = layer_num
        self.stoken_size = stoken_size
        self.use_spa = use_spa
        self.use_ssna = use_ssna
        self.use_local = use_local
        self.use_nat = use_nat
        self.use_conv = use_conv
        self.use_ffn = use_ffn
        self.use_nsp = use_nsp
        self.use_sna = use_sna
        self.use_skip = use_skip
        self.norm = LayerNorm2d(dim) if use_layer_norm else nn.Identity()
        assert use_nsp == use_sna,"Legacy name."
        # self.bn = nn.BatchNorm2d(dim)
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # -- init network for qk scale --
        # print(learn_attn_scale)
        # exit()
        _bool = (learn_attn_scale is None) or (learn_attn_scale is False)
        self.learn_attn_scale = not(_bool)
        if self.learn_attn_scale:
            attn_scale_net = AttentionScaleNet(dim, 1, ssn_nftrs)
        else:
            attn_scale_net = None

        # -- spa --
        if share_gen_sp is None:
            if spa_sim_method == "slic":
                gen_sp = GenSP(gensp_niters,M,stoken_size,affinity_softmax,
                               softmax_order,gen_sp_use_grad,gen_sp_type,
                               dim=dim,ssn_nftrs=ssn_nftrs,unet_sm=unet_sm)
            elif spa_sim_method == "bass":
                gen_sp = SimulateBass()
            else:
                raise ValueError("Uknown sim method [%s]"%spa_sim_method)
        else:
            gen_sp = share_gen_sp

        if self.use_spa:
            spa_layer = load_spa(spa_version,dim,heads,qk_dim,gen_sp,topk=topk,
                                 spa_scale=spa_scale,
                                 spa_vweight=spa_vweight,spa_oweight=spa_oweight,
                                 spa_attn_nsamples=spa_attn_nsamples,
                                 spa_attn_normz=spa_attn_normz,
                                 spa_attn_normz_nsamples=spa_attn_normz_nsamples,
                                 spa_scatter_normz=spa_scatter_normz,
                                 spa_full_sampling=spa_full_sampling,
                                 spa_sim_method=spa_sim_method,
                                 dist_type=dist_type,
                                 learn_attn_scale=attn_scale_net)
            spa_pos_enc = nn.Identity()
            # spa_pos_enc = PositionalEncodingPermute2D(dim)
        else:
            spa_layer = nn.Identity()
            spa_pos_enc = nn.Identity()
        if use_ffn: ffn_l = FFN(dim, mlp_dim, dim)
        else: ffn_l = nn.Identity()
        self.spa_layer = nn.ModuleList([spa_pos_enc,spa_layer,ffn_l])
        self.pre_layernorm = nn.Identity()
        self.post_layernorm = nn.Identity()
        self.skipped_x_shell = nn.Identity()
        self.spa_shell = nn.Identity()

        # -- neighborhood spa --
        if self.use_sna:
            sna_layer = load_sna(spa_version,dim,heads,qk_dim,gen_sp,topk=topk,
                                 spa_scale=spa_scale,
                                 spa_attn_nsamples=spa_attn_nsamples,
                                 spa_attn_normz=spa_attn_normz,
                                 spa_attn_normz_nsamples=spa_attn_normz_nsamples,
                                 spa_scatter_normz=spa_scatter_normz,
                                 spa_full_sampling=spa_full_sampling,
                                 spa_sim_method=spa_sim_method,
                                 dist_type=dist_type,kernel_size=nat_ksize,
                                 mask_labels=nsa_mask_labels,use_weights=use_weights,
                                 use_proj=use_proj,learn_attn_scale=attn_scale_net,
                                 detach_sims=detach_sims,
                                 detach_learn_attn=detach_learn_attn)
            sna_pos_enc = nn.Identity()
            # sna_pos_enc = PositionalEncodingPermute2D(dim)
        else:
            sna_layer = nn.Identity()
            sna_pos_enc = nn.Identity()
        if use_ffn: ffn_l = FFN(dim, mlp_dim, dim)
        else: ffn_l = nn.Identity()
        self.sna_layer = nn.ModuleList([sna_pos_enc,sna_layer,ffn_l])

        # -- ssna --
        if self.use_ssna:
            ssna_layer = load_ssna(spa_version,dim,heads,qk_dim,gen_sp,topk=topk,
                                   spa_scale=spa_scale,
                                   spa_vweight=spa_vweight,spa_oweight=spa_oweight,
                                   spa_attn_nsamples=spa_attn_nsamples,
                                   spa_attn_normz=spa_attn_normz,
                                   spa_attn_normz_nsamples=spa_attn_normz_nsamples,
                                   spa_scatter_normz=spa_scatter_normz,
                                   spa_full_sampling=spa_full_sampling,
                                   spa_sim_method=spa_sim_method,
                                   dist_type=dist_type,kernel_size=nat_ksize,
                                   mask_labels=nsa_mask_labels,use_weights=use_weights,
                                   use_proj=use_proj,learn_attn_scale=attn_scale_net,
                                   detach_sims=detach_sims,
                                   detach_learn_attn=detach_learn_attn)
            ssna_pos_enc = nn.Identity()
            # ssna_pos_enc = PositionalEncodingPermute2D(dim)
        else:
            ssna_layer = nn.Identity()
            ssna_pos_enc = nn.Identity()
        if use_ffn: ffn_l = FFN(dim, mlp_dim, dim)
        else: ffn_l = nn.Identity()
        self.ssna_layer = nn.ModuleList([ssna_pos_enc,ssna_layer,ffn_l])

        # -- natten --
        if self.use_nat:
            nat_layer = NeighborhoodAttention2D(dim=dim, kernel_size=nat_ksize,
                                                dilation=1, num_heads=heads,
                                                bias=False,qkv_bias=False,
                                                qk_scale=spa_scale, use_proj=use_proj,
                                                learn_attn_scale=attn_scale_net,
                                                detach_learn_attn=detach_learn_attn)
            # nat_pos_enc = PositionalEncodingPermute2D(dim)
            nat_pos_enc = nn.Identity()
        else:
            nat_pos_enc = nn.Identity()
            nat_layer = nn.Identity()
        if use_ffn: ffn_l = FFN(dim, mlp_dim, dim)
        else: ffn_l = nn.Identity()
        self.nat_layer = nn.ModuleList([nat_pos_enc,nat_layer,ffn_l])

        # -- local --
        if self.use_local:
            if use_ffn: ffn_l = FFN(dim, mlp_dim, dim)
            else: ffn_l = nn.Identity()
            self.local_layer = nn.ModuleList([
                LocalTokenAttention(dim, heads, qk_dim, stoken_size),
                ffn_l
            ])
        else:
            self.local_layer = nn.Identity()

        # -- conv --
        if self.use_conv:
            if use_ffn: ffn_l = FFN(dim, mlp_dim, dim)
            else: ffn_l = nn.Identity()
            self.conv_layer = nn.ModuleList([nn.Conv2d(dim, dim, 3, 1, 1),ffn_l])
        else:
            self.conv_layer = nn.Identity()

    def forward(self, x, state=None):

        x = self.pre_layernorm(x)
        x = self.norm(x)
        x = self.post_layernorm(x)
        if self.use_spa:
            pos_enc,sp_attn, intra_ff = self.spa_layer
            x = sp_attn(pos_enc(x)) + x
            if self.use_ffn: x = intra_ff(x) + x

        if self.use_sna:
            pos_enc,sp_attn, intra_ff = self.sna_layer
            # x = self.lrelu(x)
            x_enc = pos_enc(x)
            x = self.skipped_x_shell(x)
            sp_x = self.spa_shell(sp_attn(x_enc))
            # sp_x = self.bn(sp_x)
            # x = sp_x + x
            # x = sp_x
            # print("here: ",(sp_x-x).abs().mean().item())
            # exit()
            # x = x
            if self.use_skip:
                if self.use_ffn: x = intra_ff(sp_x) + x
                else: x = sp_x + x
            else:
                if self.use_ffn: x = intra_ff(sp_x)
                else: x = sp_x

        if self.use_ssna:
            pos_enc,sp_attn,intra_ff = self.ssna_layer
            x_enc = pos_enc(x)
            x = self.skipped_x_shell(x)
            sp_x,state = sp_attn(x_enc,state=state)
            sp_x = self.spa_shell(sp_x)
            if self.use_skip:
                if self.use_ffn: x = intra_ff(sp_x) + x
                else: x = sp_x + x
            else:
                if self.use_ffn: x = intra_ff(sp_x)
                else: x = sp_x

        if self.use_local:
            local_attn, local_ff = self.local_layer
            x = local_attn(x) + x
            if self.use_ffn: x = local_ff(x) + x

        if self.use_conv:
            conv_layer, ffn_layer = self.conv_layer
            x = conv_layer(x)
            x = ffn_layer(x)+x
            # if self.use_ffn: x = conv_ff(x) + x

        if self.use_nat:
            pos_enc,na_layer,na_ffn = self.nat_layer
            from einops import rearrange
            x = pos_enc(x)
            x = rearrange(x,'b c h w -> b h w c')
            x = na_layer(x)
            x = rearrange(x,'b h w c -> b c h w')
            if self.use_ffn: x = na_ffn(x) + x
        return x,state

