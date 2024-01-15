import math
# import torch as th
import argparse, yaml
from . import sr_utils as utils
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

# parser = argparse.ArgumentParser(description='SPIN')
# ## yaml configuration files
# parser.add_argument('--config', type=str, default=None, help = 'pre-config file for training')
# parser.add_argument('--resume', type=str, default=None, help = 'resume training or not')
# parser.add_argument('--exp-name', type=str, default=None, help = 'experiment name')
# parser.add_argument('--denoise', default=False, action="store_true")
# #parser.add_argument('--sigma', type=int, default=0., help = "sigma")

def add_noise(lr,args):
    if "sigma" in args:
        sigma = args.sigma
    else:
        sigma = 0.
    # print("lr[max,min]: ",lr.max().item(),lr.min().item())
    lr = lr + sigma*th.randn_like(lr)
    return lr

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
        "use_intra":True,"use_fnn":False,"use_nat":False,"nat_ksize":9,
        "affinity_softmax":1.,"topk":100,"intra_version":"v1",
        "data_path":"./data/sr/","data_augment":False,
        "patch_size":128,"data_repeat":1,"eval_sets":["Set5"],
        "gpu_ids":"[1]","threads":4,"model":"model",
        "decays":[],"gamma":0.5,"lr":0.0002,"resume":None,
        "log_name":"default_log","exp_name":"default_exp",
        "upscale":2,"epochs":50,"denoise":False,
        "log_every":100,"test_every":1,"batch_size":8,"sigma":25,"colors":3,
        "log_path":"output/deno/train/","resume_uuid":None,"resume_flag":False}
    for k in defs: cfg[k] = optional(cfg,k,defs[k])
    return cfg

def run(cfg):

    # -- fill missing with defaults --
    cfg = extract_defaults(cfg)
    config_via_spa(cfg)
    if cfg.denoise: cfg.upscale = 1
    resume_uuid = cfg.uuid if cfg.resume_uuid is None else cfg.resume_uuid
    if cfg.resume_flag: cfg.resume = Path(cfg.log_path) / "checkpoint" / resume_uuid
    else: cfg.resume = None

    ## set visibel gpu
    gpu_ids_str = str(cfg.gpu_ids).replace('[','').replace(']','')
    print("gpu_ids_str: ",gpu_ids_str)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)

    from dev_basics import net_chunks
    from easydict import EasyDict as edict
    import torch
    import torch as th
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import MultiStepLR, StepLR
    from superpixel_paper.sr_datas.utils import create_datasets

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

    ## definition of loss and optimizer
    loss_func = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = MultiStepLR(optimizer, milestones=cfg.decays, gamma=cfg.gamma)

    ## resume training
    start_epoch = 1
    if cfg.resume is not None:
        chkpt_files = glob.glob(os.path.join(cfg.resume, "*.ckpt"))

        if len(chkpt_files) != 0:
            chkpt_files = sorted(chkpt_files, key=lambda x: int(x.replace('.ckpt','').split('=')[-1]))
            chkpt = torch.load(chkpt_files[-1])
            prev_epoch = chkpt['epoch']
            start_epoch = prev_epoch + 1
            model.load_state_dict(chkpt['model_state_dict'])
            optimizer.load_state_dict(chkpt['optimizer_state_dict'])
            scheduler.load_state_dict(chkpt['scheduler_state_dict'])
            stat_dict = chkpt['stat_dict']
            ## reset folder and param
            # experiment_path = cfg.resume
            logging_path = os.path.join(cfg.log_path, 'logs', experiment_name)
            chkpt_path = os.path.join(cfg.log_path, 'checkpoints', experiment_name)
            log_name = os.path.join(logging_path,'log.txt')
            print('select {}, resume training from epoch {}.'.format(chkpt_files[-1], start_epoch))
    else:
        ## auto-generate the output logname
        experiment_name = cfg.uuid
        timestamp = utils.cur_timestamp_str()
        # if cfg.log_name is None:
        #     experiment_name = '{}-{}-{}-x{}-{}'.format(cfg.exp_name, cfg.model, 'fp32', cfg.upscale, timestamp)
        # else:
        #     experiment_name = '{}-{}'.format(cfg.log_name, timestamp)
        # experiment_path = os.path.join(cfg.log_path, experiment_name)
        logging_path = os.path.join(cfg.log_path, 'logs', experiment_name)
        chkpt_path = os.path.join(cfg.log_path, 'checkpoints', experiment_name)
        log_name = os.path.join(logging_path,'log.txt')
        stat_dict = utils.get_stat_dict()
        ## create folder for chkpt and stat
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

    ## start training
    timer_start = time.time()
    for epoch in range(start_epoch, cfg.nepochs+1):
        epoch_loss = 0.0
        stat_dict['epochs'] = epoch
        model = model.train()
        opt_lr = scheduler.get_last_lr()
        print('##==========={}-training, Epoch: {}, lr: {} =============##'.format('fp32', epoch, opt_lr))
        th.manual_seed(epoch)
        for iter, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            lr, hr = batch
            if cfg.denoise:
                lr = hr + cfg.sigma*th.randn_like(hr)
            lr, hr = lr.to(device), hr.to(device)
            # lr = add_noise(lr,args)
            sr = model(lr)
            eps = 1e-3
            loss = th.sqrt(th.mean((sr-hr)**2)+eps**2)
            # loss_func(sr, hr)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            if (iter + 1) % cfg.log_every == 0:
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
                # fwd_fxn = lambda vid: model(vid[:,0])[:,None]
                # fwd_fxn = net_chunks.chunk(chunk_cfg,fwd_fxn)
                # def forward(lr):
                #     with th.no_grad():
                #         sr = fwd_fxn(lr[:,None])[:,0]
                #     return sr
                avg_psnr, avg_ssim = 0.0, 0.0
                name = valid_dataloader['name']
                loader = valid_dataloader['dataloader']
                th.manual_seed(123)
                for lr, hr in tqdm(loader, ncols=80):
                    if cfg.denoise:
                        lr = hr + cfg.sigma*th.randn_like(hr)
                    lr, hr = lr.to(device), hr.to(device)
                    if cfg.topk > 600:
                        lr = lr[:,:,:256]
                        hr = hr[:,:,:cfg.upscale*256]
                    torch.cuda.empty_cache()
                    # sr = forward(lr)
                    # print("lr.shape,hr.shape: ",lr.shape,hr.shape)
                    with th.no_grad():
                        sr = model(lr)
                    # quantize output to [0, 255]
                    hr = hr.clamp(0, 255)
                    sr = sr.clamp(0, 255)
                    # conver to ycbcr
                    if cfg.colors == 3:
                        hr_ycbcr = utils.rgb_to_ycbcr(hr)
                        sr_ycbcr = utils.rgb_to_ycbcr(sr)
                        hr = hr_ycbcr[:, 0:1, :, :]
                        sr = sr_ycbcr[:, 0:1, :, :]
                    # crop image for evaluation
                    hr = hr[:, :, cfg.upscale:-cfg.upscale, cfg.upscale:-cfg.upscale]
                    sr = sr[:, :, cfg.upscale:-cfg.upscale, cfg.upscale:-cfg.upscale]
                    # calculate psnr and ssim
                    psnr = utils.calc_psnr(sr, hr)
                    ssim = utils.calc_ssim(sr, hr)
                    # psnr = utils.calc_psnr(sr, hr)
                    # ssim = utils.calc_ssim(sr, hr)

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
