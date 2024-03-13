import math
# import torch as th
import argparse, yaml
# from . import sr_utils as utils
from superpixel_paper.deno_trte import sr_utils as utils
import os
from tqdm import tqdm
import logging
import sys
import time
import importlib
import glob
import copy
dcopy = copy.deepcopy
from easydict import EasyDict as edict
from pathlib import Path
from ..spa_config import config_via_spa
from superpixel_paper.utils import hooks
from torchvision.utils import save_image,make_grid

# import torch as th
# th.autograd.set_detect_anomaly(True)

# parser = argparse.ArgumentParser(description='SPIN')
# ## yaml configuration files
# parser.add_argument('--config', type=str, default=None, help = 'pre-config file for training')
# parser.add_argument('--resume', type=str, default=None, help = 'resume training or not')
# parser.add_argument('--exp-name', type=str, default=None, help = 'experiment name')
# parser.add_argument('--denoise', default=False, action="store_true")
# #parser.add_argument('--sigma', type=int, default=0., help = "sigma")

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    # torch.autograd.set_detect_anomaly(True)

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def add_noise(lr,args):
    if "sigma" in args:
        sigma = args.sigma
    else:
        sigma = 0.
    # print("lr[max,min]: ",lr.max().item(),lr.min().item())
    lr = lr + sigma*th.randn_like(lr)
    return lr

def epoch_from_chkpt(ckpt_dir):
    import torch

    if not Path(ckpt_dir).exists():
        return -1
    chkpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if len(chkpt_files) == 0: return -1
    chkpt_files = sorted(chkpt_files, key=lambda x: int(x.replace('.ckpt','').split('=')[-1]))
    chkpt = torch.load(chkpt_files[-1])
    prev_epoch = chkpt['epoch']
    return prev_epoch

# if __name__ == '__main__':
#     print("PID: ",os.getpid())
#     args = parser.parse_args()
#     if args.config:
#        opt = vars(args)
#        yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
#        opt.update(yaml_args)
#     if args.denoise: args.upscale = 1

#     ## set visibel gpu
#     gpu_ids_str = str(args.gpu_ids).replace('[','').replace(']','')
#     print("gpu_ids_str: ",gpu_ids_str)
#     os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#     os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)

#     from dev_basics import net_chunks
#     from easydict import EasyDict as edict
#     import torch
#     import torch.nn as nn
#     import torch.nn.functional as F
#     from torch.optim.lr_scheduler import MultiStepLR, StepLR
#     from spin.datas.utils import create_datasets

#     # -- chunking for validation --
#     chunk_cfg = edict()
#     chunk_cfg.spatial_chunk_size = 256
#     chunk_cfg.spatial_chunk_overlap = 0.1

def optional(pydict,key,default):
    if pydict is None: return default
    elif key in pydict: return pydict[key]
    else: return default

def extract_defaults(_cfg):
    cfg = edict(dcopy(_cfg))
    defs = {
        "dim":12,"qk_dim":6,"mlp_dim":6,"stoken_size":[8],"block_num":1,
        "heads":1,"M":0.,"use_local":False,"use_inter":False,
        "use_intra":True,"use_ffn":False,"use_nat":False,"nat_ksize":9,
        "affinity_softmax":1.,"topk":100,"intra_version":"v1",
        "data_path":"./data/sr/","data_augment":False,
        "patch_size":128,"data_repeat":4,"eval_sets":["Set5"],
        "gpu_ids":"[1]","threads":4,"model_name":"simple",
        "decays":[],"gamma":0.5,"lr":0.0002,"resume":None,
        "log_name":"default_log","exp_name":"default_exp",
        "upscale":1,"epochs":50,"denoise":False,
        "log_every":100,"test_every":1,"batch_size":8,"sigma":25,"colors":3,
        "log_path":"output/deno/train/",
        "resume_uuid":None,"resume_flag":True,"resume_weights_only":False,
        "spatial_chunk_size":256,"spatial_chunk_overlap":0.25,
        "gradient_clip":0.,"train_only_attn_scale":False,
        "ssn_loss":False,"ssn_loss_lamb":0.1,"use_dataparallel":True,
        "deno_loss_lamb":1.,"ssn_target":"seg",
    }
    for k in defs: cfg[k] = optional(cfg,k,defs[k])
    return cfg

def load_model(cfg):
    try:
        if cfg.model_name == "simple":
            import_str = 'superpixel_paper.models.model'
        elif cfg.model_name == "nlrn":
            import_str = 'superpixel_paper.nlrn.model'
        else:
            raise ValueError("")
        model = utils.import_module(import_str).create_model(cfg)
    except Exception:
        raise ValueError('not supported model type! or something')
    return model

def get_resume_ckpt(cfg):
    resume_uuid = cfg.uuid if cfg.resume_uuid is None else cfg.resume_uuid
    resume_epoch = -1
    if cfg.resume_flag:
        dir0 = Path(cfg.log_path) / "checkpoints" / resume_uuid
        dir1 = Path(cfg.log_path) / "checkpoints" / cfg.uuid
        epoch0 = epoch_from_chkpt(dir0)
        epoch1 = epoch_from_chkpt(dir1)
        if epoch0 >= epoch1:
            resume = dir0
            resume_epoch = epoch0
        else:
            resume = dir1
            resume_epoch = epoch1
    else:
        resume = None
        resume_epoch = -1
    if (not(resume is None) and not(resume.exists())) or (resume_epoch < 0):
        resume = None
    return resume

def run(cfg):

    # -- fill missing with defaults --
    cfg = extract_defaults(cfg)
    config_via_spa(cfg)

    ## set visibel gpu
    # gpu_ids_str = str(cfg.gpu_ids).replace('[','').replace(']','')
    # print("gpu_ids_str: ",gpu_ids_str)
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)
    seed_everything(cfg.seed)

    from dev_basics import net_chunks
    from easydict import EasyDict as edict
    import torch
    import torch as th
    import torch.nn as nn
    import torch.nn.functional as F
    from einops import rearrange
    from torch.optim.lr_scheduler import MultiStepLR, StepLR
    from superpixel_paper.sr_datas.utils import create_datasets
    from superpixel_paper.models.utils import set_train_mode
    from superpixel_paper.models.utils import train_only_learn_attn_scale

    # -- resume --
    cfg.resume = get_resume_ckpt(cfg)

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
    from superpixel_paper.sr_datas.utils import get_seg_dataset
    train_dataloader, _ = get_seg_dataset(cfg)
    _, valid_dataloaders = create_datasets(cfg)

    ## definitions of model
    model = load_model(cfg)
    print(model)
    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    print('#Params : {:<.4f} [K]'.format(num_parameters / 10 ** 3))
    from superpixel_paper.models.sp_hooks import SsnaSuperpixelHooks
    if cfg.use_dataparallel:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    if cfg.ssn_loss:
        sphooks = SsnaSuperpixelHooks(model)
    else:
        sphooks = None
    # exit()

    ## definition of loss and optimizer
    loss_func = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # cfg.decays = [27,] + cfg.decays
    scheduler = MultiStepLR(optimizer, milestones=cfg.decays, gamma=cfg.gamma)
    # set_train_mode(model,False,False)
    # set_train_mode(model,True,False)


    ## resume training
    start_epoch = 1
    if cfg.resume is not None:
        chkpt_files = glob.glob(os.path.join(cfg.resume, "*.ckpt"))
        if len(chkpt_files) != 0:
            chkpt_files = sorted(chkpt_files, key=lambda x: int(x.replace('.ckpt','').split('=')[-1]))
            experiment_name = cfg.uuid
            logging_path = os.path.join(cfg.log_path, 'logs', experiment_name)
            chkpt_path = os.path.join(cfg.log_path, 'checkpoints', experiment_name)
            log_name = os.path.join(logging_path,'log.txt')
            chkpt = torch.load(chkpt_files[-1])
            prev_epoch = 0 if cfg.resume_weights_only else chkpt['epoch']
            start_epoch = prev_epoch + 1
            model.load_state_dict(chkpt['model_state_dict'])
            if not(cfg.resume_weights_only):
                optimizer.load_state_dict(chkpt['optimizer_state_dict'])
                stat_dict = chkpt['stat_dict']
            else:
                stat_dict = utils.get_stat_dict()
            print("Resuming from ",chkpt_files[-1])
            print('select {}, resume training from epoch {}.'\
                  .format(chkpt_files[-1], start_epoch))
        else:
            print("error.")
            exit()
    else:
        ## auto-generate the output logname
        experiment_name = cfg.uuid
        timestamp = utils.cur_timestamp_str()
        logging_path = os.path.join(cfg.log_path, 'logs', experiment_name)
        chkpt_path = os.path.join(cfg.log_path, 'checkpoints', experiment_name)
        log_name = os.path.join(logging_path,'log.txt')
        stat_dict = utils.get_stat_dict()
        ## create folder for chkpt and stat
    # exit()

    # -- init log --
    if not os.path.exists(logging_path): os.makedirs(logging_path)
    if not os.path.exists(chkpt_path): os.makedirs(chkpt_path)
    # experiment_model_path = os.path.join(experiment_path, 'checkpoints')
    # if not os.path.exists(experiment_model_path):
    #     os.makedirs(experiment_model_path)
    ## save training paramters
    exp_params = vars(cfg)
    exp_params_name = os.path.join(logging_path,'config.yml')
    with open(exp_params_name, 'w') as exp_params_file:
        yaml.dump(exp_params, exp_params_file, default_flow_style=False)

    ## print architecture of model
    time.sleep(3) # sleep 3 seconds
    sys.stdout = utils.ExperimentLogger(log_name, sys.stdout)
    print(model)
    # exit()
    sys.stdout.flush()

    if cfg.train_only_attn_scale:
        print("training only attention scale.")
        train_only_learn_attn_scale(model)
        # for name,param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        # exit()

    # -- chunking for validation --
    chunk_cfg = edict()
    chunk_cfg.spatial_chunk_sr = cfg.upscale
    chunk_cfg.spatial_chunk_size = cfg.spatial_chunk_size
    chunk_cfg.spatial_chunk_overlap = cfg.spatial_chunk_overlap

    from superpixel_paper.ssn_trte.label_loss import SuperpixelLoss
    loss_type = "cross" if cfg.ssn_target == "seg" else "mse"
    sp_loss_fxn = SuperpixelLoss(loss_type)

    swap_mode = False
    # if swap_mode:
    #     print("\n"*3)
    #     print("Swap mode training.")

    ## start training
    timer_start = time.time()
    for epoch in range(start_epoch, cfg.nepochs+1):
        epoch_loss = 0.0
        stat_dict['epochs'] = epoch
        model = model.train()
        # hook = hooks.AttentionHook(model)
        opt_lr = scheduler.get_last_lr()
        print('##==========={}-training, Epoch: {}, lr: {} =============##'.format('fp32', epoch, opt_lr))
        th.manual_seed(int(cfg.seed)+epoch)

        #     set_train_mode(model,(epoch%10)<5,(epoch%10)>=5)
        # if cfg.use_alt_train:
        # if epoch >= 100:
        #     if epoch == 100: print("swap on.")
        #     set_train_mode(model,(epoch%10)<5,(epoch%10)>=5)
        # elif (epoch >= 50) and (epoch < 100):
        #     if (epoch == 50): print("only lambda_at.")
        #     set_train_mode(model,True,False)
        # else:
        #     if (epoch == 1): print("only sims")
        #     set_train_mode(model,False,True)
        for iter, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            img, seg = batch
            img, seg = img.to(device), seg.to(device)
            noisy = img + cfg.sigma*th.randn_like(img)

            deno = model(noisy)
            # print(deno.shape)

            eps = 1e-3
            loss = 0
            if cfg.deno_loss_lamb > 1e-10:
                loss = th.sqrt(th.mean((deno-img)**2)+eps**2)
                loss = cfg.deno_loss_lamb * loss
            if cfg.ssn_loss and (cfg.ssn_loss_lamb > 1e-10):
                B = noisy.shape[0]
                HW = deno.shape[-2]*deno.shape[-1]
                sims = sphooks.spix[0]
                sims = sims.reshape(B,HW,-1).transpose(2,1)
                # print("sims.shape: ",sims.shape,seg.shape)
                if cfg.ssn_target == "seg":
                    ssn_loss = sp_loss_fxn(seg[:,None],sims)
                elif cfg.ssn_target == "pix":
                    ssn_loss = sp_loss_fxn(img,sims)
                else:
                    raise ValueError(f"Uknown target [{cfg.ssn_target}]")
                # print(float(ssn_loss))
                loss = loss + cfg.ssn_loss_lamb * ssn_loss

            if (swap_mode) and (epoch >= 25):
                set_train_mode(model,(iter%10)<5,(iter%10)>=5)

            loss.backward()
            if cfg.gradient_clip > 0:
                clip = cfg.gradient_clip
                th.nn.utils.clip_grad_norm_(model.parameters(),clip)
            # vgrad = hooks.get_qkv_grads(model)
            # print(vgrad.shape)
            # exit()
            optimizer.step()
            epoch_loss += float(loss)
            if (cfg.batch_size*(iter + 1)) % cfg.log_every == 0:
                cur_steps = (iter + 1) * cfg.batch_size
                total_steps = len(train_dataloader.dataset)
                fill_width = math.ceil(math.log10(total_steps))
                cur_steps = str(cur_steps).zfill(fill_width)

                epoch_width = math.ceil(math.log10(cfg.nepochs))
                cur_epoch = str(epoch).zfill(epoch_width)

                avg_loss = epoch_loss / (iter + 1)
                stat_dict['losses'].append(avg_loss)

                timer_end = time.time()
                duration = timer_end - timer_start
                timer_start = timer_end
                print('Epoch:{}, {}/{}, loss: {:.4f}, time: {:.3f}'.format(cur_epoch, cur_steps, total_steps, avg_loss, duration))
            # break

        # import pdb; pdb.set_trace()
        if epoch % cfg.test_every == 0:
            torch.cuda.empty_cache()
            torch.set_grad_enabled(False)
            test_log = ''
            model = model.eval()
            for valid_dataloader in valid_dataloaders:
                fwd_fxn = lambda vid: model(vid[:,0])[:,None]
                fwd_fxn = net_chunks.chunk(chunk_cfg,fwd_fxn)
                def chunk_forward(lr):
                    sr = fwd_fxn(lr[:,None])[:,0]
                    return sr
                avg_psnr, avg_ssim = 0.0, 0.0
                name = valid_dataloader['name']
                loader = valid_dataloader['dataloader']
                th.manual_seed(123)

                for img, _tmp in tqdm(loader, ncols=80):
                    img = img.to(device)
                    noisy = img + cfg.sigma*th.randn_like(img)
                    # print("img.shape: ",img.shape,_tmp.shape)
                    # print(lr.shape,hr.shape)
                    # if cfg.topk > 600:
                    if noisy.shape[-1] == 228: # just a specific example
                        noisy = noisy[...,:224]
                        img = img[...,:224]
                    torch.cuda.empty_cache()
                    # sr = forward(lr)
                    # print("lr.shape,hr.shape: ",lr.shape,hr.shape)
                    with th.no_grad():
                        if (cfg.topk > 500) or (cfg.model_name == "nlrn"):
                            deno = chunk_forward(noisy)
                        else:
                            deno = model(noisy)
                    # quantize output to [0, 255]
                    img = img.clamp(0, 255)
                    deno = deno.clamp(0, 255)

                    # conver to ycbcr
                    if cfg.colors == 3:
                        img_ycbcr = utils.rgb_to_ycbcr(img)
                        deno_ycbcr = utils.rgb_to_ycbcr(deno)
                        img = img_ycbcr[:, 0:1, :, :]
                        deno = deno_ycbcr[:, 0:1, :, :]
                    # crop image for evaluation
                    img = img[:, :, cfg.upscale:-cfg.upscale, cfg.upscale:-cfg.upscale]
                    deno = deno[:, :, cfg.upscale:-cfg.upscale, cfg.upscale:-cfg.upscale]
                    # calculate psnr and ssim
                    psnr = utils.calc_psnr(deno,img)
                    ssim = utils.calc_ssim(deno,img)

                    avg_psnr += psnr
                    avg_ssim += ssim
                avg_psnr = round(avg_psnr/len(loader) + 5e-3, 2)
                avg_ssim = round(avg_ssim/len(loader) + 5e-5, 4)
                stat_dict[name]['psnrs'].append(avg_psnr)
                stat_dict[name]['ssims'].append(avg_ssim)
                # print(avg_psnr)
                # exit()

                if stat_dict[name]['best_ssim']['value'] > 0.98:
                    stat_dict[name]['best_ssim']['value'] = avg_ssim
                    stat_dict[name]['best_ssim']['epoch'] = epoch

                if stat_dict[name]['best_psnr']['value'] < avg_psnr:
                    stat_dict[name]['best_psnr']['value'] = avg_psnr
                    stat_dict[name]['best_psnr']['epoch'] = epoch
                if stat_dict[name]['best_ssim']['value'] < avg_ssim:
                    stat_dict[name]['best_ssim']['value'] = avg_ssim
                    stat_dict[name]['best_ssim']['epoch'] = epoch
                test_log += '[{}-X{}], PSNR/SSIM: {:.2f}/{:.4f} \
                (Best: {:.2f}/{:.4f}, Epoch: {}/{})\n'.format(name, cfg.upscale,\
                    float(avg_psnr), float(avg_ssim),stat_dict[name]['best_psnr']['value'], stat_dict[name]['best_ssim']['value'],stat_dict[name]['best_psnr']['epoch'], stat_dict[name]['best_ssim']['epoch'])

            print(test_log)
            sys.stdout.flush()
            # save model
            model_str = '%s-epoch=%02d.ckpt'%(cfg.uuid,epoch-1) # "start at 0"
            saved_model_path = os.path.join(chkpt_path,model_str)
            # torch.save(model.state_dict(), saved_model_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'stat_dict': stat_dict
            }, saved_model_path)
            torch.set_grad_enabled(True)
            # save stat dict
            ## save training paramters
            # stat_dict_name = os.path.join(experiment_path, 'stat_dict.yml')
            stat_dict_name = os.path.join(logging_path, 'stat_dict_%d.yml' % epoch)
            with open(stat_dict_name, 'w') as stat_dict_file:
                yaml.dump(stat_dict, stat_dict_file, default_flow_style=False)

        ## update scheduler
        scheduler.step()

    # -- return info --
    info = edict()
    # print(stat_dict)
    for valid_dataloader in valid_dataloaders:
        name = valid_dataloader['name']
        info["%s_best_psnrs"%name] = stat_dict[name]['best_psnr']['value']
        info["%s_best_ssim"%name] = stat_dict[name]['best_ssim']['value']
    return info
