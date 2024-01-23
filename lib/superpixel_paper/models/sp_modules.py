
# -- pytorch --
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# -- basics --
import math
from einops import rearrange

# -- superpixel --
from superpixel_paper.est_attn_normz import EffNormzFunction
from spin.models.pair_wise_distance import PairwiseDistFunction
from natten import NeighborhoodAttention2D


class GenSP(nn.Module):
    def __init__(self, n_iter=2,M=0.,stoken_size=8,
                 affinity_softmax=1., softmax_order="v0", use_grad=False):
        super().__init__()
        self.n_iter = n_iter
        self.M = M
        self.stoken_size = stoken_size
        self.affinity_softmax = affinity_softmax
        self.softmax_order = softmax_order
        self.use_grad = use_grad

    def forward(self, x):
        soft_association, num_spixels = ssn_iter(x, self.stoken_size,
                                                 self.n_iter, self.M,
                                                 self.affinity_softmax,
                                                 self.softmax_order,
                                                 self.use_grad
        )
        return soft_association, num_spixels

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


def ssn_iter(pixel_features, stoken_size=[16, 16], n_iter=2, M = 0.,
             affinity_softmax=1., softmax_order="v0", use_grad=False):
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

    # -- add grid --
    from stnls.dev.slic.utils import append_grid,add_grid
    # print("pixel_features.shape: ",pixel_features.shape,M/stoken_size[0])
    # pixel_features = append_grid(pixel_features[:,None],M/stoken_size[0])[:,0]
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

    with torch.set_grad_enabled(use_grad):
        for k in range(n_iter):

            # -- compute all affinities  --
            dist_matrix = PairwiseDistFunction.apply(
                    pixel_features, spixel_features, init_label_map,
                num_spixels_width, num_spixels_height)
            # print("dist_matrix.shape: ",dist_matrix.shape)
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

    return abs_affinity, num_spixels



