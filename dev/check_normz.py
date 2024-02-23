
"""

   Verify the normalization of the expected value's lower-bound
   does not sum to 1 (and thus needs correcting.)

"""
import torch as th

def main():

    W = 11
    dists_raw = th.rand(W**2)
    dists_raw[0] = 2
    dists_raw[3] = 2
    dists = th.exp(dists_raw)
    energy = th.rand((W**2,9))
    energy[0] = 1
    energy[1] = 1
    probs = th.nn.functional.softmax(2*energy,1)
    use_lb = True
    i = 0
    weights = th.zeros((W**2))
    for s in range(9):
        for j in range(W**2):

            # -- num --
            if use_lb:
                num = probs[i,s] * ( (i == j) + probs[j,s]*(i != j) ) * dists[j]
            else:
                num = probs[i,s] * probs[j,s] * dists[j]

            # -- denom --
            denom = 0
            for jp in range(W**2):
                if ((jp == i) or (jp == j)) and use_lb:
                    denom += dists[jp]
                else:
                    denom += probs[jp,s] * dists[jp]

            weights[j] += num / denom
    # print(weights,weights.sum())
    print(weights.sum())


if __name__ == "__main__":
    main()
