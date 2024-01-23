
import torch as th
from natten import NeighAttnMat,NeighAttnAgg
from natten import NeighborhoodAttention2D
from natten import natten_padding,natten_remove_padding
from superpixel_paper.nsp.nsp_agg import NeighSuperpixelAgg,NeighSuperpixelAggFunction
from superpixel_paper.nsp import NeighborhoodSuperpixelAttention
from einops import rearrange,repeat


# -- bench --
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt


# -- settings --
dim = 3
nat_ksize = 5
heads = 1
bias = False

device = "cuda"
B = 2
# H,W = 16,16
H,W = 32,32
# H,W = 32,48
# H,W = 64,96
# H,W = 128,128
img = 10.*th.rand((B,H,W,dim)).to(device)


# -- init --
nat_ours = NeighborhoodSuperpixelAttention(dim=dim, kernel_size=nat_ksize,
                                           num_heads=heads,qkv_bias=bias,
                                           bias=bias,qk_scale=1.).to(device)
nat_full = NeighborhoodAttention2D(dim=dim, kernel_size=nat_ksize,
                                   num_heads=heads,qkv_bias=bias,
                                   bias=bias,qk_scale=1.).to(device)
nat_mat = NeighAttnMat(dim=dim, kernel_size=nat_ksize, num_heads=heads,
                       qk_bias=bias,bias=bias).to(device)
nat_agg = NeighAttnAgg(dim=dim,num_heads=heads,kernel_size=nat_ksize,
                       v_bias=bias).to(device)
nat_mat.qk.weight.data[...] = nat_full.qkv.weight.data[:2*dim]
nat_agg.v.weight.data[...] = nat_full.qkv.weight.data[2*dim:]
nat_agg.proj.weight.data[...] = nat_full.proj.weight.data[...]
nat_agg.proj.bias.data[...] = nat_full.proj.bias.data[...]

nat_ours.nat_mat.qk.weight.data[...] = nat_full.qkv.weight.data[:2*dim]
nat_ours.nat_agg.v.weight.data[...] = nat_full.qkv.weight.data[2*dim:]
nat_ours.nat_agg.proj.weight.data[...] = nat_full.proj.weight.data[...]
nat_ours.nat_agg.proj.bias.data[...] = nat_full.proj.bias.data[...]

def check_fxn(img,attn,kernel_size):
    B,H,W,C = img.shape
    img_pad,ipad = natten_padding(img,kernel_size)
    nat_agg.v.weight.data = nat_agg.v.weight.data.type(img.dtype)
    nat_agg.proj.weight.data = nat_agg.proj.weight.data.type(img.dtype)
    vals = (nat_agg.v(img_pad).
            reshape(B, H, W, 1, nat_agg.num_heads, nat_agg.head_dim).
            permute(3, 0, 4, 1, 2, 5))[0]
    attn = attn.softmax(-1)
    imgSp = th.zeros_like(vals[:,0,:,:,0]).long()
    out = NeighSuperpixelAggFunction.apply(vals,attn,imgSp)
    out = out.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
    out = natten_remove_padding(out,ipad)
    out = nat_agg.proj(out)
    return out

# print(nat_agg(img,nat_mat(img)).shape)
# print(nat_full(img).shape)

# out0 = nat_agg(img,nat_mat(img))
# attn0 = nat_mat(img)
# out1,attn1 = nat_full(img)
# print(th.mean((attn0 - attn1)**2))

img2 = img.clone().requires_grad_(True)
attn2 = nat_mat(img2).requires_grad_(True)
# out2 = check_fxn(img,nat_mat(img),nat_ksize)
out2 = check_fxn(img2,attn2,nat_ksize)

img0 = img.clone().requires_grad_(True)
attn0 = nat_mat(img0).requires_grad_(True)
out0 = nat_agg(img0,attn0)

# labels = th.zeros_like(img[:,:,:,0]).long()
# img3 = img.clone().requires_grad_(True)
# attn3 = nat_ours.nat_mat(img3,labels)

img3 = img.clone().requires_grad_(True)
dir0,dir1 = 'b h w c -> b c h w','b c h w -> b h w c'
labels = th.zeros_like(img3[:,:,:,0]).long()
# print(labels.shape)
out3 = rearrange(nat_ours(rearrange(img3,dir0),labels),dir1)
print("out3.shape: ",out3.shape)
print("out0.shape: ",out0.shape)
out1 = nat_full(img)

# print("attn error: ",th.mean((attn2-attn0)**2).item())
# exit()
# print(attn3.shape)

print("-"*20)
print("ours")
print("-"*20)
print(out3[0,:8,:8,0])
print(out3[0,-8:,-8:,0])
# print(attn3[0,0,:8,:8,0])
print("-"*20)
print("theirs")
print("-"*20)
# print(attn0[0,0,:8,:8,0])
print(out0[0,:8,:8,0])
print(out0[0,-8:,-8:,0])

print("Errors")
print(th.mean((out0 - out2)**2))
print(th.mean((out0 - out1)**2))
print(th.mean((out0 - out3)**2))
# exit()


loss3 = out3.abs().mean()
loss2 = out2.abs().mean()
loss0 = out0.abs().mean()
attn0.retain_grad()
attn2.retain_grad()
img0.retain_grad()
img2.retain_grad()
img3.retain_grad()
loss3.backward()
loss2.backward()
loss0.backward()
# attn0.abs().mean().backward()
# attn.abs().mean().backward()
# attn2.abs().mean().backward()

print(attn0.grad[0,0,:3,:3,:2])
print(attn2.grad[0,0,:3,:3,:2])
# print(th.sort(attn0.grad.ravel()).values)
# print(th.sort(attn2.grad.ravel()).values)

eps = 1e-6
print("[d_attn error]: ",th.mean(th.abs(attn0.grad-attn2.grad)/(attn0.grad.abs()+eps)))

print(th.sort(img0.grad.ravel()).values)
print(th.sort(img2.grad.ravel()).values)

print("-"*20)
print("theirs")
print("-"*20)
# print(img0.grad[0,:3,:3,:2])
print(img0.grad[0,-3:,-3:,:2])
# print(img2.grad[0,:3,:3,:2])
print("-"*20)
print("ours")
print("-"*20)
print(img3.grad[0,:3,:3,:2])
print(img3.grad[0,-3:,-3:,:2])

print("[d_img error(0,2)]: ",th.mean(th.abs(img0.grad-img2.grad)/(img0.grad.abs()+eps)))
print("[d_img error(0,2)]: ",th.max(th.abs(img0.grad-img2.grad)/(img0.grad.abs()+eps)))
print("[d_img error(0,3)]: ",th.mean(th.abs(img0.grad-img3.grad)/(img0.grad.abs()+eps)))
print("[d_img error(0,3)]: ",th.max(th.abs(img0.grad-img3.grad)/(img0.grad.abs()+eps)))
# exit()

# -- init bench --
timer,memer = ExpTimer(),GpuMemer()

# -- ours --
img2 = img.clone().requires_grad_(True)
attn2 = nat_mat(img2)
with TimeIt(timer,"fwd"):
    with MemIt(memer,"fwd"):
        out2 = check_fxn(img2,attn2,nat_ksize)
loss2 = out2.abs().mean()
with TimeIt(timer,"bwd"):
    with MemIt(memer,"bwd"):
        loss2.backward()

# -- theirs --
img0 = img.clone().requires_grad_(True)
attn0 = nat_mat(img0)
with TimeIt(timer,"fwd_theirs"):
    with MemIt(memer,"fwd_theirs"):
        out0 = nat_agg(img0,attn0)
loss0 = out0.abs().mean()
with TimeIt(timer,"bwd_theirs"):
    with MemIt(memer,"bwd_theirs"):
        loss0.backward()

print(timer)
print(memer)


# exit()

# img2 = img.clone().requires_grad_(True).double()
# attn2 = attn2.double()
# fwd_fxn0 = lambda x: check_fxn(x,attn2,nat_ksize)
# th.autograd.gradcheck(fwd_fxn0, img2, eps=1e-5,
#                       atol=1e-5, nondet_tol=1e-5, raise_exception=True)
# attn = attn2.clone().double()
# img = img.clone().double()
# fwd_fxn0 = lambda x: check_fxn(img,x,nat_ksize)
# th.autograd.gradcheck(fwd_fxn0, attn, eps=1e-5,
#                       atol=1e-5, nondet_tol=1e-5, raise_exception=True)


# from superpixel_paper/models/
# from
# gensp = GenSP(n_iter=2,M=0.,stoken_size=8,
#               affinity_softmax=1., softmax_order="v0", use_grad=False)
nat_ours = NeighborhoodSuperpixelAttention(dim=dim, kernel_size=nat_ksize,
                                           num_heads=heads,qkv_bias=bias,
                                           bias=bias,qk_scale=1.).to(device).double()
B,H,W,_ = img.shape
labels = th.randint(0, 3, (B,H,W)).to(device).long()
print(labels)
def check_fxn_ours(x):
    dir0,dir1 = 'b h w c -> b c h w','b c h w -> b h w c'
    out = rearrange(nat_ours(rearrange(x,dir0),labels),dir1)
    return out
img = img.clone().double().requires_grad_(True)
th.autograd.gradcheck(check_fxn_ours, img, eps=1e-5,
                      atol=1e-5, nondet_tol=1e-5, raise_exception=True)

