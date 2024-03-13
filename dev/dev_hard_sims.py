"""

Convert probs to 1,0 with "1" at max

"""

import torch as th
from torch.nn.functional import one_hot


sims = th.rand(3,4).round(decimals=3)
sims = sims/sims.sum(-1,keepdim=True)
print(sims)
print(sims.shape)
# inds = th.max(sims,dim=-1,keepdim=True).indices
inds = sims.argmax(-1)
sims = one_hot(inds,sims.shape[-1])
# inds = sims.argmax(-1)[:,None]
# sims[...] = 0.
# sims = sims.scatter(1,inds,1)
print(sims)
print(sims.shape)
