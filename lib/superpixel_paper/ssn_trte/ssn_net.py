
import torch
import torch as th
import torch.nn as nn
from skimage import io, color
from superpixel_paper.models.sp_modules import ssn_iter
from superpixel_paper.models.ssn_model import UNet as SsnNet


def rgb2lab(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to Lab.

    Args:
        image: RGB Image to be converted to Lab with shape :math:`(*, 3, H, W)`.

    Returns:
        Lab version of the image with shape :math:`(*, 3, H, W)`.
        The L channel values are in the range 0..100. a and b are in the range -128..127.
    """

    # Convert from sRGB to Linear RGB
    lin_rgb = torch.where(image > 0.04045,
                          torch.pow(((image + 0.055) / 1.055), 2.4), image / 12.92)
    r,g,b = lin_rgb[..., 0,:,:],lin_rgb[..., 1,:,:],lin_rgb[..., 2,:,:]
    x = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b
    xyz_im = torch.stack([x, y, z], -3)

    # normalize for D65 white point
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883],device=xyz_im.device,
                                 dtype=xyz_im.dtype)[None, :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    threshold = 0.008856
    power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int = torch.where(xyz_normalized > threshold, power, scale)

    x,y,z = xyz_int[..., 0,:,:],xyz_int[..., 1,:,:],xyz_int[..., 2,:,:]
    L = (116.0 * y) - 16.0
    a = 500.0 * (x - y)
    _b = 200.0 * (y - z)
    out = torch.stack([L, a, _b], dim=-3)

    return out

class UNetSsnNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, init_features=6,
                 softmax=True,n_iters=5, stoken_size=8, softmax_scale=1, M = 0.1):
        super(UNetSsnNet, self).__init__()
        self.softmax = softmax
        self.ftrs = SsnNet(in_channels,out_channels,init_features)
        self.n_iters = n_iters
        self.stoken_size = stoken_size
        self.softmax_scale = softmax_scale
        self.M = M
        # print(self.n_iters,self.stoken_size,self.softmax_scale,self.M)

    def forward(self, x):
        ftr = self.ftrs(x)
        # lab = rgb2lab(x)
        # print(lab)
        ftrs = th.cat([ftr,x],-3)
        stoken = [self.stoken_size,self.stoken_size]
        _,sims,_ = ssn_iter(ftrs, M = self.M,
                            stoken_size=stoken,n_iter=self.n_iters,
                            affinity_softmax=self.softmax_scale,
                            use_grad=True)
        return sims

