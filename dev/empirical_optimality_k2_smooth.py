
import numpy as np
import torch as th
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class EstSSNA(nn.Module):
    def __init__(self,C2,W):
        super().__init__()
        self.C2 = C2
        self.energy = self.init_energy(C2,W)
        self.weights = nn.Parameter(th.ones(W))

    def init_energy(self,C2,W):
        energy = th.zeros((W,2)).cuda()
        energy[:C2,0] = 1
        energy[:C2,1] = 10

        energy[C2:,0] = 10
        energy[C2:,1] = 1

        return energy

    def compute_weights(self,mean,sigma2):
        # -- compute probs --
        center = -1
        probs = th.softmax(self.weights[:,None]*self.energy,1)
        probs0 = probs[center,0] * probs[:,0]/th.sum(probs[:,0]) # sum over j
        probs1 = probs[center,1] * probs[:,1]/th.sum(probs[:,1]) # sum over j
        weights = probs0 + probs1 # sum over "s"
        return weights

    def forward(self,mean,sigma2):

        # -- compute error --
        weights = self.compute_weights(mean,sigma2)
        bias = mean**2 * (th.sum(weights[:self.C2]**2))
        var = sigma2 * (th.sum(weights**2))
        error = bias + var
        return error

def get_error_na(mean,sigma2,C2,W):
    weights = np.ones((W))/W
    bias = mean**2 * (np.sum(weights[:C2]**2))
    var = sigma2 * (np.sum(weights**2))
    error = bias + var
    return error

def est_error_sna(mean,sigma2,C2,W):
    weights = np.ones((W))
    weights[:C2] = 0
    weights /= weights.sum()
    bias = mean**2 * (np.sum(weights[:C2]**2))
    var = sigma2 * (np.sum(weights**2))
    # print("sna: ",bias,var,sigma2)
    error = bias + var
    return error

def est_error_ssna(mean,sigma2,C2,W):
    model = EstSSNA(C2,W).cuda()
    model = model.train()
    optimizer = optim.Adam(model.parameters(),1e-2)
    error_prev = None
    niters = 200
    # niters = 500
    # niters = 50
    # niters = 1
    # w0 = model.weights.clone()
    # w_og = w0.clone()
    # print(model.compute_weights(mean,sigma2))

    for _ in range(niters):
        optimizer.zero_grad()
        model.zero_grad()

        error = model(mean,sigma2)
        error.backward()
        # print(model.weights.grad)
        # print(model.weights,w0)
        # print(th.mean((w0 - model.weights)**2))
        # w0 = model.weights.clone()
        optimizer.step()
        # if error < 1e-10: break
        # if error_prev is None: error_prev = error.item()
        # elif th.abs(error - error_prev)<1e-10: break

    error = model(mean,sigma2).detach().item()
    # print(model.compute_weights(mean,sigma2))
    # exit()
    return error

def main():

    # -- create --
    # W = 9*9
    W = 5*5
    N = 30
    # N = 10
    means = np.linspace(0.0,1,N)[::-1]
    sigma2 = (np.linspace(0.,50,N)/255.)**2
    # sigma2 = sigma2[::-1]
    # wgrid = [int(0.5*W)]
    wgrid = [int(.25*W),int(0.5*W),int(0.75*W)]
    # wgrid = [int(.5*W)]#,int(0.5*W),int(0.75*W)]
    # wgrid = np.linspace(0,W,3)
    probs_na = np.zeros((N,N))
    probs_sna = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            # for k in range(int(.25*W),W,3):
            # for k in range(int(.25*W),int(0.75*W),2):
            # for k in [int(.25*W),int(0.5*W),int(0.75*W)]:
            for k in wgrid:
                error_na = get_error_na(means[i],sigma2[j],k,W)
                error_ssna = est_error_ssna(means[i],sigma2[j],k,W)
                error_sna = est_error_sna(means[i],sigma2[j],k,W)
                error_na = round(error_na,8)
                error_sna = round(error_sna,8)
                error_ssna = round(error_ssna,8)
                print(i,j,means[i],sigma2[j],k,error_na,error_sna,error_ssna)
                # exit()
                # probs_na[i,j] += (error_na > error_ssna)
                # probs_sna[i,j] += (error_sna > error_ssna)
                probs_na[i,j] += error_na - error_ssna
                probs_sna[i,j] += error_sna - error_ssna

                # exit()
            probs_na[i,j] /= len(wgrid)
            probs_sna[i,j] /= len(wgrid)

    # print(probs_na.max(),probs_sna.max())
    # print(probs_na.min(),probs_sna.min())
    vmin = min([probs_na.min(),probs_sna.min()])
    vmax = max([probs_na.max(),probs_sna.max()])

    # -- plot --
    dpi = 200
    ginfo = {'wspace':0.25, 'hspace':0.,
             "top":0.98,"bottom":0.09,"left":.12,"right":0.82}
    fig,ax = plt.subplots(1,2,figsize=(6.,2.75),gridspec_kw=ginfo,dpi=200)
    ax[0].imshow(probs_na,vmin=vmin, vmax=vmax, cmap='winter')#, aspect='auto')
    im2 = ax[1].imshow(probs_sna,vmin=vmin, vmax=vmax, cmap='winter')#, aspect='auto')

    # -- format --
    # ax[0].set_title(r"$P\left[ \hat{R}(f_{SSNA}) < R(f_{NA})\right]$")
    # ax[1].set_title(r"$P\left[ \hat{R}(f_{SSNA}) < R(f_{SNA})\right]$")
    ax[0].set_title(r"$R(f_{NA}) - \hat{R}(f_{SSNA})$")
    ax[1].set_title(r"$R(f_{SNA}) - \hat{R}(f_{SSNA})$")


    Nplt = min(len(sigma2),5)
    # print("-"*5)
    # print(0,len(sigma2),Nplt)
    inds = np.linspace(0,len(sigma2)-1,Nplt).round().astype(np.int)
    # sigma_sel = (np.array([0,15,25,35,50])/255.)**2
    # inds = []
    # for sig in sigma_sel:
    #     inds.append(np.where(sigma2 == sig)[0])
    # print(inds)
    # print(sigma2)
    # print(means)
    # xticks = [0,15,25,35,50]
    xticks = sigma2[inds]
    ax[0].set_xticks(inds)
    ax[1].set_xticks(inds)
    xticks = np.round(np.sqrt(xticks)*255.,0)
    ax[0].set_xticklabels(["%d" % w for w in xticks],fontsize=12)
    ax[1].set_xticklabels(["%d" % w for w in xticks],fontsize=12)

    inds = np.linspace(0,len(sigma2)-1,Nplt).round().astype(np.int)
    yticks = means[inds]
    ax[0].set_yticks(inds)
    yticks = np.round(yticks,1)
    ax[0].set_yticklabels(["%.1f" % w for w in yticks],fontsize=12)
    ax[1].set_yticks(inds)
    ax[1].set_yticklabels([])


    ax[0].set_ylabel("$(\mu_1 - \mu_2)^2$",fontsize=12)
    ax[0].set_xlabel("$\sigma_N^2$",fontsize=12)
    ax[1].set_xlabel("$\sigma_N^2$",fontsize=12)

    # -- colorbar --
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    # cbar.set_label('title', rotation=0)

    # -- save --
    plt.savefig("output/figures/empirical_optimality_k2_smooth_diff.png",
                transparent=False)


if __name__ == "__main__":
    main()
