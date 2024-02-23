
import numpy as np
dims = [5,10,9]
vals = np.zeros(list(dims))

num = dims[0] * dims[1] * dims[2]
for i in range(num):
    i0 = i // (dims[1] * dims[2])
    tmp = i - i0 * (dims[1] * dims[2])
    i1 = tmp // dims[2]
    i2 = tmp - i1 * dims[2]
    # int ibatch = blockIdx.z / (nheads*nspix);
    # int tmp = blockIdx.z - ibatch * (nheads * nspix);
    # int ihead = tmp/nspix;
    # int si = tmp - ihead * nspix;
    # print(i0,i1,i2)
    vals[i0,i1,i2] += 1
assert np.all(vals == 1)
print("all good :D")
