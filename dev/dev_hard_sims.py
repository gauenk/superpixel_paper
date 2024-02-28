"""

Convert probs to 1,0 with "1" at max

"""

import torch as th

sims = th.rand(3,4).round(decimals=3)
sims = sims/sims.sum(-1,keepdim=True)
inds = th.max(sims,dim=-1,keepdim=True).indices
sims = sims.scatter(1,inds,1)
print(sims)
