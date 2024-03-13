import torch as th
import math
import argparse, yaml
from superpixel_paper.utils import spin_utils as utils
# import spin.utils as utils
import os
from tqdm import tqdm
import logging
import sys
import time
import importlib
import glob
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict
from dev_basics.utils.misc import set_seed
from dev_basics.utils.metrics import compute_psnrs,compute_ssims

import copy
dcopy = copy.deepcopy
from easydict import EasyDict as edict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
# from spin.datas.utils import create_datasets
from superpixel_paper.sr_datas.utils import create_datasets
# from .share import config_via_spa
from ..spa_config import config_via_spa


# parser = argparse.ArgumentParser(description='SPIN')
# ## yaml configuration files
# parser.add_argument('--config', type=str, default=None, help = 'pre-config file for training')
# parser.add_argument('--resume', type=str, default=None, help = 'resume training or not')
# # parser.add_argument('--task', type=str, default="sr", help = 'which task?')
# parser.add_argument('--save_to_tmp', default=False, action="store_true")

# if __name__ == '__main__':
#     args = parser.parse_args()
#     if args.config:
#        opt = vars(args)
#        yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
#        opt.update(yaml_args)
#     opt = edict(opt)

def allows_sna(state):
    new_state = dcopy(state)
    for key,val in state.items():
        if "nsp" in key:
            key_new = dcopy(key).replace("nsp","sna")
            # print(key_new,key)
            new_state[key_new] = new_state[key].clone()
            del new_state[key]
    return new_state

def optional(pydict,key,default):
    if pydict is None: return default
    elif key in pydict: return pydict[key]
    else: return default

def extract_defaults(_cfg):
    cfg = edict(dcopy(_cfg))
    defs = {
        "dim":12,"qk_dim":6,"mlp_dim":6,"stoken_size":[8],"block_num":1,
        "heads":1,"M":0.,"use_local":False,"use_inter":False,
        "use_intra":True,"use_fnn":False,"use_nat":False,"nat_ksize":9,
        "affinity_softmax":1.,"topk":100,"intra_version":"v1",
        "data_path":"./data/sr/","data_augment":False,
        "patch_size":128,"data_repeat":1,"eval_sets":["Set5"],
        "gpu_ids":"[1]","threads":4,"model":"model",
        "decays":[],"gamma":0.5,"lr":0.0002,"resume":None,
        "log_name":"default_log","exp_name":"default_exp",
        "upscale":2,"epochs":50,"denoise":False,
        "log_every":100,"test_every":1,"batch_size":8,"sigma":25,"colors":3,
        "log_path":"output/deno/train/","resume_uuid":None,"resume_flag":True,
        "output_folder":"output/deno/test","save_output":False,"eval_ycb":True}
    # defs = {
    #     "dim":12,"qk_dim":6,"mlp_dim":6,"stoken_size":[8],"block_num":1,
    #     "heads":1,"M":0.,"use_local":False,"use_inter":False,
    #     "use_intra":True,"use_fnn":False,"use_nat":False,"nat_ksize":9,
    #     "affinity_softmax":1.,"topk":100,"intra_version":"v1",
    #     "data_path":"./data/sr/","data_augment":False,"seed":123,
    #     "patch_size":128,"data_repeat":1,"eval_sets":["Set5"],
    #     "gpu_ids":"[0]","threads":4,"model":"model",
    #     "decays":[],"gamma":0.5,"lr":0.0001,"resume":None,
    #     "log_name":"default_log","exp_name":"default_exp",
    #     "upscale":2,"epochs":50,"denoise":False,
    #     "log_every":100,"test_every":1,"batch_size":8,"sigma":25,"colors":3,
    #     "log_path":"output/deno/train/","resume_uuid":None,"resume_flag":True,
    #     "output_folder":"output/deno/test","save_output":False}
    for k in defs: cfg[k] = optional(cfg,k,defs[k])
    return cfg

def run(cfg):

    # -- fill missing with defaults --
    cfg = extract_defaults(cfg)
    config_via_spa(cfg)
    if cfg.denoise: cfg.upscale = 1
    resume_uuid = cfg.tr_uuid if cfg.resume_uuid is None else cfg.resume_uuid
    if cfg.resume_flag: cfg.resume = Path(cfg.log_path) / "checkpoints" / resume_uuid
    else: cfg.resume = None
    set_seed(cfg.seed)
    ## set visibel gpu
    # gpu_ids_str = str(cfg.gpu_ids).replace('[','').replace(']','')
    # gpu_ids_str = "2"
    # # gpu_ids_str = "0"
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)
    test_epoch = int(cfg.pretrained_path.split("=")[-1].split(".")[0])
    # out_base = "%d/%s/epoch=%02d" %(cfg.tr_uuid,test_epoch)
    out_base = "%d/%s/epoch=%02d" %(cfg.sigma,cfg.tr_uuid[:5],test_epoch)
    output_folder = Path(cfg.output_folder) / out_base
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    from dev_basics import net_chunks
    from easydict import EasyDict as edict

    # -- chunking for validation --
    chunk_cfg = edict()
    chunk_cfg.spatial_chunk_size = 256
    chunk_cfg.spatial_chunk_sr = cfg.upscale
    chunk_cfg.spatial_chunk_overlap = 0.25

    ## select active gpu devices
    device = None
    if cfg.gpu_ids is not None and torch.cuda.is_available():
        print('use cuda & cudnn for acceleration!')
        print('the gpu id is: {}'.format(cfg.gpu_ids))
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        print('use cpu for training!')
        device = torch.device('cpu')
    torch.set_num_threads(cfg.threads)

    ## create dataset for training and validating
    train_dataloader, valid_dataloaders = create_datasets(cfg)

    ## definitions of model
    try:
        import_str = 'superpixel_paper.models.{}'.format(cfg.model)
        model = utils.import_module(import_str).create_model(cfg)
    except Exception:
        raise ValueError('not supported model type! or something')
    model = nn.DataParallel(model).to(device)

    ## resume training
    start_epoch = 1
    assert cfg.resume is not None
    chkpt_files = glob.glob(os.path.join(cfg.resume, "*.ckpt"))
    # print(cfg.resume)
    # print(chkpt_files)
    # if len(chkpt_files) != 0:
    #     chkpt_files = sorted(chkpt_files, key=lambda x: int(x.replace('.ckpt','').split('=')[-1]))
    #     chkpt_fn = chkpt_files[-1]
    chkpt_fn = os.path.join(cfg.resume,cfg.pretrained_path)
    print("Checkpoint: ",chkpt_fn)
    ckpt = torch.load(chkpt_fn)
    prev_epoch = ckpt['epoch']

    # mstate = cleanup_mstate(ckpt['model_state_dict'])
    state = ckpt['model_state_dict']
    state = allows_sna(state)
    model.load_state_dict(state)
    stat_dict = ckpt['stat_dict']
    print('select {} for testing'.format(chkpt_fn))
    init_keys = list(stat_dict.keys())

    ## print architecture of model
    time.sleep(3) # sleep 3 seconds
    print(model)

    epoch = 1
    torch.set_grad_enabled(False)
    test_log = ''
    model = model.eval()
    info = {"dname":[],"name":[],"psnrs":[],"ssims":[]}
    for valid_dataloader in valid_dataloaders:
        fwd_fxn = lambda vid: model(vid[:,0])[:,None]
        # def fwd_fxn(vid):
        #     sr = torch.nn.functional.interpolate(vid[0], scale_factor=2,
        #                                          mode='bilinear', align_corners=False)
        #     return sr[:,None]
        fwd_fxn = net_chunks.chunk(chunk_cfg,fwd_fxn)
        def forward(_sample):
            with th.no_grad():
               sr = fwd_fxn(_sample[:,None])[:,0]
            # run = lambda fxn: fxn(sr).item()
            # print("[min,max,mean]: ",run(th.min),run(th.max),run(th.mean))
            # print("sr,lr [shapes]: ",sr.shape,lr.shape)
            return sr
        avg_psnr, avg_ssim = 0.0, 0.0
        name = valid_dataloader['name']
        loader = valid_dataloader['dataloader']
        data_i = valid_dataloader['data'] if "data" in valid_dataloader else None
        count = 0
        for lr, hr in tqdm(loader, ncols=80):
            count += 1
            if cfg.denoise:
                lr = hr + cfg.sigma*th.randn_like(hr)
            lr, hr = lr.to(device), hr.to(device)
            # print("lr.shape: ",lr.shape)
            torch.cuda.empty_cache()
            # lr = lr[...,:300,:300]
            # hr = hr[...,:300*cfg.upscale,:300*cfg.upscale]
            # print("lr.shape,hr.shape: ",lr.shape,hr.shape)
            # print("sigma: ",th.mean((lr/255. - hr/255.)**2).sqrt()*255.)
            with th.no_grad():
                sr = forward(lr)
                # sr = model(lr)
            # quantize output to [0, 255]
            hr = hr.clamp(0, 255)
            sr = sr.clamp(0, 255)
            # print("hr.shape: ",hr.shape,sr.shape,lr.shape)
            # exit()

            out_img = sr.detach()[0].float().cpu().numpy()
            out_img = np.transpose(out_img, (1, 2, 0))

            output_name = os.path.join(str(output_folder), str(name))
            if not os.path.exists(output_name):
                os.makedirs(output_name)
            output_name = os.path.join(output_name,
                                         str(count) + '_x' + str(cfg.upscale) + '.png')

            if cfg.save_output:
                print(output_name)
                cv2.imwrite(output_name, out_img[:, :, [2, 1, 0]]) #

            # conver to ycbcr

            hr = hr[:, :, cfg.upscale:-cfg.upscale, cfg.upscale:-cfg.upscale]
            sr = sr[:, :, cfg.upscale:-cfg.upscale, cfg.upscale:-cfg.upscale]

            if (cfg.colors == 3) and cfg.eval_ycb:
                hr_ycbcr = utils.rgb_to_ycbcr(hr)
                sr_ycbcr = utils.rgb_to_ycbcr(sr)
                hr = hr_ycbcr[:, 0:1, :, :]
                sr = sr_ycbcr[:, 0:1, :, :]
                psnr = utils.calc_psnr(sr, hr)
                ssim = utils.calc_ssim(sr, hr)
            else:
                psnr = compute_psnrs(hr[:,None],sr[:,None],div=255.)[0].item()
                ssim = compute_ssims(hr[:,None],sr[:,None],div=255.)[0].item()
            avg_psnr += psnr
            avg_ssim += ssim

            # print("-"*20)
            if not(data_i is None):
                info['dname'].append(name)
                info['name'].append(Path(data_i.lr_filenames[count-1]).stem)
                info['psnrs'].append(psnr)
                info['ssims'].append(ssim)
                # print(data_i.lr_filenames[count-1],psnr,ssim)

        avg_psnr = round(avg_psnr/len(loader) + 5e-3, 2)
        avg_ssim = round(avg_ssim/len(loader) + 5e-5, 4)

        if not(name in stat_dict):
            subkeys = ["psnrs","ssims"]
            stat_dict[name] = {}
            for k in subkeys: stat_dict[name][k] = []
            subkeys = ["best_psnr","best_ssim"]
            for k in subkeys: stat_dict[name][k] = {"value":0}

        stat_dict[name]['psnrs'].append(avg_psnr)
        stat_dict[name]['ssims'].append(avg_ssim)
        if stat_dict[name]['best_psnr']['value'] < avg_psnr:
            stat_dict[name]['best_psnr']['value'] = avg_psnr
            stat_dict[name]['best_psnr']['epoch'] = epoch
        if stat_dict[name]['best_ssim']['value'] < avg_ssim:
            stat_dict[name]['best_ssim']['value'] = avg_ssim
            stat_dict[name]['best_ssim']['epoch'] = epoch
        test_log += '[{}-X{}], PSNR/SSIM: {:.2f}/{:.4f} (Best: {:.2f}/{:.4f}, Epoch: {}/{})\n'.format(name, cfg.upscale, float(avg_psnr), float(avg_ssim), 
            stat_dict[name]['best_psnr']['value'], stat_dict[name]['best_ssim']['value'], 
            stat_dict[name]['best_psnr']['epoch'], stat_dict[name]['best_ssim']['epoch'])

    print(test_log)
    info = pd.DataFrame(info)
    for dname,df in info.groupby("dname"):
        print("----- %s -----"%dname)
        print(df)
    # print(opt,cfg)
    # # if cfg.save_to_tmp:
    # #     info.to_csv(".tmp/results.csv",index=False)
    # print(info.to_dict(orient="records"))
    # exit()
    info = info.rename(columns={"name":"iname"})

    return info.to_dict(orient="records")


def cleanup_mstate(mstate):
    states = {}
    for key,state in mstate.items():
        if "cached_penc" in key: continue
        states[key] = state
    return states
