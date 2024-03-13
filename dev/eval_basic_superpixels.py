
# -- basic --
import numpy as np
from scipy.io import loadmat
import torch as th

# -- [reference method] superpixel eval --
import seal

# -- superpixel eval --
from superpixel_paper.utils import metrics,extract_defaults
from superpixel_paper.utils.connected import connected_sp

def main():

    # -- read segmentation --
    name = "2018"
    gt_fn = "data/sr/BSD500/groundTruth/test/%s.mat" % name
    annos = loadmat(gt_fn)['groundTruth']
    seg = annos[0][0]['Segmentation'][0][0]
    print(seg)

    # -- read superpixel --
    fn = "../BASS/pytorch_version_og/out_sp200_s123/csv/%s.csv" % name
    spix = np.genfromtxt(fn, delimiter=",").astype(int)
    spix = connected_sp(spix,0.5,3)
    print(spix)

    # -- misc --
    h,w = spix.shape
    label_list = spix.astype(int).flatten().tolist()
    gtseg_list = seg.astype(int).flatten().tolist()
    asa = seal.computeASA(label_list, gtseg_list, 0)
    br = seal.computeBR(label_list, gtseg_list, h, w, 1)
    print(asa,br)

    # -- compute asa/br --
    asa = metrics.compute_asa(spix,seg)
    br = metrics.compute_br(spix,seg,r=1)
    bp = metrics.compute_bp(spix,seg,r=1)
    print(asa,br,bp)

if __name__ == "__main__":
    main()
