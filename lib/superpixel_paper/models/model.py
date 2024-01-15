
# -- import torch --
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# -- basic --
import math
from einops import rearrange
from easydict import EasyDict as edict

# -- submodules --
from .share import extract
from .spa_menu import load_spa
from .sp_modules import GenSP
from .bass import SimulateBass
from .guts import LocalTokenAttention,FFN,LayerNorm2d


def create_model(args):
    args = edict(vars(args))
    pairs = {"topk":None,"use_local":False,"use_spa":True,
             "use_nat":False,"nat_ksize":7,"use_conv":False,
             "softmax_order":"v0","base_first":False,"spa_version":"v1",
             "use_ffn":False,"spa_scale":None,
             "spa_kweight":True,"spa_oweight":True,
             "spa_attn_nsamples":10,
             "spa_attn_normz":False,"spa_attn_normz_nsamples":10,
             "spa_scatter_normz":False,"spa_full_sampling":False}
    extract(args,pairs)
    print({k:args[k] for k in pairs})
    return SimpleModel(colors=args.colors, dim=args.dim, block_num=args.block_num,
                       heads=args.heads, qk_dim=args.qk_dim,
                       mlp_dim=args.mlp_dim, stoken_size=args.stoken_size,
                       upscale=args.upscale, M=args.M, use_local=args.use_local,
                       use_spa=args.use_spa,
                       use_nat=args.use_nat,nat_ksize=args.nat_ksize,
                       use_conv=args.use_conv,
                       affinity_softmax=args.affinity_softmax,topk=args.topk,
                       softmax_order=args.softmax_order,base_first=args.base_first,
                       spa_version=args.spa_version,use_ffn=args.use_ffn,
                       spa_kweight=args.spa_kweight,
                       spa_oweight=args.spa_oweight,spa_scale=args.spa_scale,
                       spa_attn_nsamples=args.spa_attn_nsamples,
                       spa_attn_normz=args.spa_attn_normz,
                       spa_attn_normz_nsamples=args.spa_attn_normz_nsamples,
                       spa_scatter_normz=args.spa_scatter_normz,
                       spa_full_sampling=args.spa_full_sampling)

class SimpleModel(nn.Module):
    def __init__(self, colors=3, dim=40, block_num=8, heads=1, qk_dim=24, mlp_dim=72,
                 stoken_size=[12, 16, 20, 24, 12, 16, 20, 24], upscale=3,
                 M=0., use_local=True, use_spa=True,
                 use_nat=False, nat_ksize=7, use_conv=False,
                 affinity_softmax=1.,topk=None,softmax_order="v0",
                 base_first=False,spa_version="v1",use_ffn=True,
                 spa_kweight=True,spa_oweight=True,spa_scale=None,
                 spa_attn_nsamples=10,
                 spa_attn_normz=None,spa_attn_normz_nsamples=10,
                 spa_normz_nsamples=10,spa_normz_version="map",
                 spa_scatter_normz=None,spa_full_sampling=False):
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
        self.first_conv = nn.Conv2d(colors, dim, 3, 1, 1)

        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        for i in range(block_num):
            self.blocks.append(Block(dim=dim, layer_num=2,
                                     stoken_size=[stoken_size[i], stoken_size[i]],
                                     heads=heads, qk_dim=qk_dim, mlp_dim=mlp_dim,
                                     M=M, use_local=use_local,
                                     use_spa=use_spa,
                                     use_nat=use_nat,nat_ksize=nat_ksize,
                                     use_conv=use_conv,
                                     affinity_softmax=affinity_softmax,topk=topk,
                                     softmax_order=softmax_order,
                                     spa_version=spa_version,use_ffn=use_ffn,
                                     spa_kweight=spa_kweight,spa_oweight=spa_oweight,
                                     spa_scale=spa_scale,
                                     spa_attn_nsamples=spa_attn_nsamples,
                                     spa_attn_normz=spa_attn_normz,
                                     spa_attn_normz_nsamples=spa_attn_normz_nsamples,
                                     spa_scatter_normz=spa_scatter_normz,
                                     spa_full_sampling=spa_full_sampling))
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
            msg = 'Upscale factor is expected to be one'
            msg += '  of (2, 3, 4), but got {}'.format(upscale)
            raise NotImplementedError(msg)
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
        x = x / 255.

        if self.upscale != 1:
            base = torch.nn.functional.interpolate(x, scale_factor=self.upscale,
                                                   mode='bilinear', align_corners=False)
        else:
            base = x

        if self.base_first: x = base
        x = self.first_conv(x)

        for i in range(self.block_num):
            residual = self.blocks[i](x)
            x = x + self.mid_convs[i](residual)

        if self.upscale == 4:
            out = self.pixel_shuffle(self.upconv1(x))
            out = self.pixel_shuffle(self.upconv2(out))
        else:
            out = self.pixel_shuffle(self.upconv(x))
        out = self.lrelu(out)
        out = base + self.last_conv(out)

        return out * 255.

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
                 M=0., use_local=False, use_spa=True,
                 use_nat=False, nat_ksize=7, use_conv=False,
                 affinity_softmax=1., topk=None, softmax_order="v0",
                 spa_version="v1",use_ffn=True,spa_scale=None,
                 spa_kweight=True,spa_oweight=True,
                 spa_attn_nsamples=10,spa_attn_normz=None,
                 spa_attn_normz_nsamples=10,spa_scatter_normz=None,
                 spa_full_sampling=False,spa_sim_method="slic"):
        super(Block,self).__init__()
        self.layer_num = layer_num
        self.stoken_size = stoken_size
        self.use_spa = use_spa
        self.use_local = use_local
        self.use_nat = use_nat
        self.use_conv = use_conv
        self.use_ffn = use_ffn
        self.norm = LayerNorm2d(dim)

        # -- spa --
        if spa_sim_method == "slic":
            gen_sp = GenSP(3,M,stoken_size,affinity_softmax,softmax_order)
        elif spa_sim_method == "bass":
            gen_sp = SimulateBass()
        else:
            raise ValueError("Uknown sim method [%s]"%spa_sim_method)
        spa_layer = load_spa(spa_version,dim,heads,qk_dim,gen_sp,topk=topk,
                             spa_scale=spa_scale,
                             spa_kweight=spa_kweight,spa_oweight=spa_oweight,
                             spa_attn_nsamples=spa_attn_nsamples,
                             spa_attn_normz=spa_attn_normz,
                             spa_attn_normz_nsamples=spa_attn_normz_nsamples,
                             spa_scatter_normz=spa_scatter_normz,
                             spa_full_sampling=spa_full_sampling,
                             spa_sim_method=spa_sim_method)
        if use_ffn: ffn_l = FFN(dim, mlp_dim, dim)
        else: ffn_l = nn.Identity()
        self.spa_layer = nn.ModuleList([spa_layer,ffn_l])

        # -- natten --
        if self.use_nat:
            self.nat_layer = NeighborhoodAttention2D(dim=dim, kernel_size=nat_ksize,
                                                     dilation=1, num_heads=heads)
        else:
            self.nat_layer = nn.Identity()

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
            self.conv_layer = nn.Conv2d(dim, dim, 3, 1, 1)
        else:
            self.conv_layer = nn.Identity()

    def forward(self, x):

        x = self.norm(x)
        if self.use_spa:
            sp_attn, intra_ff = self.spa_layer
            x = sp_attn(x) + x
            if self.use_ffn: x = intra_ff(x) + x

        if self.use_local:
            local_attn, local_ff = self.local_layer
            x = local_attn(x) + x
            if self.use_ffn: x = local_ff(x) + x

        if self.use_conv:
            x = self.conv_layer(x)

        if self.use_nat:
            from einops import rearrange
            x = rearrange(x,'b c h w -> b h w c')
            x = self.nat_layer(x)
            x = rearrange(x,'b h w c -> b c h w')

        return x
