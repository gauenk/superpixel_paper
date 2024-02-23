import os
# from spin.datas.benchmark import Benchmark
# from spin.datas.div2k import DIV2K
from .benchmark import Benchmark
from .div2k import DIV2K
from .bsd500 import BSD500
from torch.utils.data import DataLoader


def create_datasets(args):

    if not("dname_tr" in args) or (args.dname_tr == "div2k"):
        div2k = DIV2K(
            os.path.join(args.data_path, 'DIV2K/DIV2K_train_HR'), 
            os.path.join(args.data_path, 'DIV2K/DIV2K_train_LR_bicubic'), 
            os.path.join(args.data_path, 'div2k_cache'),
            train=True, 
            augment=args.data_augment, 
            upscale=args.upscale, 
            colors=args.colors, 
            patch_size=args.patch_size, 
            repeat=args.data_repeat, 
        )
        train_dataloader = DataLoader(dataset=div2k, num_workers=args.threads,
                                      batch_size=args.batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
    elif args.dname_tr == "bsd500":
        bsd500 = BSD500(
            os.path.join(args.data_path, 'BSD500/BSD500_train_HR'), 
            os.path.join(args.data_path, 'BSD500/BSD500_train_LR_bicubic'), 
            os.path.join(args.data_path, 'bsd500_cache'),
            train=True, 
            augment=args.data_augment, 
            upscale=args.upscale, 
            colors=args.colors, 
            patch_size=args.patch_size, 
            repeat=args.data_repeat,
            img_postfix=".jpg"
        )
        train_dataloader = DataLoader(dataset=bsd500, num_workers=args.threads,
                                      batch_size=args.batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
    else:
        raise ValueError("Unknown dataset: %s" % args.dname_tr)

    valid_dataloaders = []
    if 'BSD68' in args.eval_sets:
        bsd68_hr_path = os.path.join(args.data_path, 'benchmarks/BSD68/HR')
        bsd68_lr_path = os.path.join(args.data_path, 'benchmarks/BSD68/LR_bicubic')
        bsd68  = Benchmark(bsd68_hr_path, bsd68_lr_path, scale=args.upscale,
                          colors=args.colors)
        valid_dataloaders += [{'name': 'bsd68', 'dataloader':
                               DataLoader(dataset=bsd68, batch_size=1, shuffle=False),
                               "data":bsd68}]
    if 'Set5' in args.eval_sets:
        set5_hr_path = os.path.join(args.data_path, 'benchmarks/Set5/HR')
        set5_lr_path = os.path.join(args.data_path, 'benchmarks/Set5/LR_bicubic')
        set5  = Benchmark(set5_hr_path, set5_lr_path, scale=args.upscale,
                          colors=args.colors)
        valid_dataloaders += [{'name': 'set5', 'dataloader':
                               DataLoader(dataset=set5, batch_size=1, shuffle=False),
                               "data":set5}]
    if 'Set8' in args.eval_sets:
        set8_hr_path = os.path.join(args.data_path, 'benchmarks/Set8/HR')
        set8_lr_path = os.path.join(args.data_path, 'benchmarks/Set8/LR_bicubic')
        set8  = Benchmark(set8_hr_path, set8_lr_path, scale=args.upscale,
                          colors=args.colors)
        valid_dataloaders += [{'name': 'set8', 'dataloader':
                               DataLoader(dataset=set8, batch_size=1, shuffle=False),
                               "data":set8}]
    if 'iPhoneSum2023' in args.eval_sets:
        ips2023_hr_path = os.path.join(args.data_path, 'benchmarks/iPhoneSum2023/HR')
        ips2023_lr_path = os.path.join(args.data_path, 'benchmarks/iPhoneSum2023/LR_bicubic')
        ips2023  = Benchmark(ips2023_hr_path, ips2023_lr_path, scale=args.upscale,
                             colors=args.colors)
        valid_dataloaders += [{'name': 'ips2023', 'dataloader':
                               DataLoader(dataset=ips2023, batch_size=1, shuffle=False),
                               "data":ips2023}]
    if 'Set14' in args.eval_sets:
        set14_hr_path = os.path.join(args.data_path, 'benchmarks/Set14/HR')
        set14_lr_path = os.path.join(args.data_path, 'benchmarks/Set14/LR_bicubic')
        set14 = Benchmark(set14_hr_path, set14_lr_path,
                          scale=args.upscale, colors=args.colors)
        valid_dataloaders += [{'name': 'set14', 'dataloader':
                               DataLoader(dataset=set14, batch_size=1, shuffle=False),
                               "data":set14}]
    if 'B100' in args.eval_sets:
        b100_hr_path = os.path.join(args.data_path, 'benchmarks/B100/HR')
        b100_lr_path = os.path.join(args.data_path, 'benchmarks/B100/LR_bicubic')
        b100  = Benchmark(b100_hr_path, b100_lr_path, scale=args.upscale,
                          colors=args.colors)
        valid_dataloaders += [{'name': 'b100', 'dataloader':
                               DataLoader(dataset=b100, batch_size=1, shuffle=False),
                               "data":b100}]
    if 'Urban100' in args.eval_sets:
        u100_hr_path = os.path.join(args.data_path, 'benchmarks/Urban100/HR')
        u100_lr_path = os.path.join(args.data_path, 'benchmarks/Urban100/LR_bicubic')
        u100  = Benchmark(u100_hr_path, u100_lr_path, scale=args.upscale,
                          colors=args.colors)
        valid_dataloaders += [{'name': 'u100', 'dataloader':
                               DataLoader(dataset=u100, batch_size=1, shuffle=False),
                               "data":u100}]
    if 'Manga109' in args.eval_sets:
        manga_hr_path = os.path.join(args.data_path, 'benchmarks/Manga109/HR')
        manga_lr_path = os.path.join(args.data_path, 'benchmarks/Manga109/LR_bicubic')
        manga = Benchmark(manga_hr_path, manga_lr_path, scale=args.upscale,
                          colors=args.colors)
        valid_dataloaders += [{'name': 'manga109', 'dataloader':
                               DataLoader(dataset=manga, batch_size=1, shuffle=False),
                               "data":manga}]
    if len(valid_dataloaders) == 0:
        print('select no dataset for evaluation!')
    else:
        selected = ''
        for i in range(len(valid_dataloaders)):
            selected += ", " + valid_dataloaders[i]['name']
        print('select {} for evaluation! '.format(selected))
    return train_dataloader, valid_dataloaders
