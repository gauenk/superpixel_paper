"""

   Example:

   my_fxn = lambda vid: stnls.nn.another_fxn(vid)
   ana = stnls.testing.gradcheck.get_ana_jacobian(my_fxn,vid)
   num = stnls.testing.gradcheck.get_num_jacobian(my_fxn,vid,eps=1e-2,nreps=1)
   assert th.abs(ana-num).mean()<1e-3

"""

import torch
import torch as th

def get_num_jacobian(fxn,inputs,eps=1e-3,nreps=1):
    from torch.autograd.gradcheck import _get_numerical_jacobian
    num = _get_numerical_jacobian(fxn, (inputs,),
                                  eps=eps, is_forward_ad=False)[0][0]
    for i in range(nreps-1):
        num += get_num_jacobian(fxn,inputs,eps=eps)
    num /= nreps
    return num

def get_ana_jacobian(fxn,inputs,eps=1e-5):
    from torch.autograd.gradcheck import _check_analytical_jacobian_attributes
    out = fxn(inputs)
    ana = _check_analytical_jacobian_attributes((inputs,), out, eps, False)[0]
    return ana

def get_gradcheck_pair(fxn,inputs,eps=1e-3):
    num = get_num_jacobian(fxn,inputs,eps=1e-3)
    ana = get_ana_jacobian(fxn,inputs)
    return num,ana

def gradcheck_skip_nan_unstable(fxn, inputs, rtol=1e-05, atol=1e-08,
                                nreps=3, num_eps=5e-4, unstable_eps=1e-2):
    num = get_num_jacobian_skip_unstable(fxn,inputs,eps=num_eps,
                                         nreps=nreps,unstable_eps=unstable_eps)
    ana = get_ana_jacobian(fxn,inputs)
    args = th.where(th.logical_and(~th.isnan(num),num.abs()>0))
    args1 = th.where(th.abs(num[args]-ana[args])>1e-2)[0]
    # print("ana: ",ana[47,573:575])
    # print(num[:5,:5])
    # print(ana[:5,:5])
    # print(num[-5:,-5:])
    # print(ana[-5:,-5:])
    # # print(num.shape)
    # print(num[args][args1][:20])
    # print(ana[args][args1][:20])
    # print([args[i][args1] for i in range(2)])
    return th.allclose(num[args],ana[args],atol=atol,rtol=rtol)

def gradcheck_skipnan(fxn,inputs, rtol=1e-05, atol=1e-08, nreps=1, num_eps=5e-4):
    num = get_num_jacobian(fxn,inputs,eps=num_eps,nreps=nreps)
    ana = get_ana_jacobian(fxn,inputs)
    args = th.where(th.logical_and(~th.isnan(num),num.abs()>0))
    args1 = th.where(th.abs(num[args]-ana[args])>5e-1)[0]
    # print(num[:7,:7])
    # print(ana[:7,:7])
    # print("-"*20)
    # print(num[-7:,-7:])
    # print(ana[-7:,-7:])
    # print("-"*20)
    # # print("-"*20)
    # print(num[args][args1][:20])
    # print(ana[args][args1][:20])
    # print([args[i][args1] for i in range(2)])
    return th.allclose(num[args],ana[args],atol=atol,rtol=rtol)

def get_num_jacobian_skip_unstable(fxn,inputs,eps=1e-3,nreps=1,unstable_eps=1e-2):
    from torch.autograd.gradcheck import _get_numerical_jacobian
    nums = []
    for i in range(nreps):
        eps_i = eps * (1 + i*eps)
        num = _get_numerical_jacobian(fxn, (inputs,),
                                      eps=eps_i, is_forward_ad=False)[0][0]
        nums.append(num)

    delta = th.zeros_like(nums[0])
    for i in range(nreps):
        # print(nums[i][47,573:575])
        for j in range(nreps):
            if i >= j: continue
            # print(i,j)
            delta += th.abs(nums[i] - nums[j])
    # print(delta)
    # print(delta[~th.isnan(delta)].min(),delta[~th.isnan(delta)].max())
    # print("Percentage unstable: ",100*th.mean(1.*(delta > unstable_eps)).item())
    unstable = th.where(th.logical_or(delta > unstable_eps,th.isnan(delta)))
    num = th.mean(th.stack(nums),dim=0)
    num[unstable] = th.nan
    # print(num)
    # print(nums[0])
    return num

