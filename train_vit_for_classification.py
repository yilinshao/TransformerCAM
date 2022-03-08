# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import math
import shutil
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as torch_dist

from core.puzzle_utils import *
from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from tqdm import tqdm
from baselines.ViT.ViT_new import vit_base_patch16_224

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_epoch', default=15, type=int)

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--min_image_size', default=180, type=int)
parser.add_argument('--max_image_size', default=260, type=int)

parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='', type=str)

# For Puzzle-CAM
parser.add_argument('--num_pieces', default=4, type=int)

# 'cl_pcl'
# 'cl_re'
# 'cl_conf'
# 'cl_pcl_re'
# 'cl_pcl_re_conf'
parser.add_argument('--loss_option', default='cl_pcl_re', type=str)

parser.add_argument('--level', default='feature', type=str)

parser.add_argument('--re_loss', default='L1_Loss', type=str)  # 'L1_Loss', 'L2_Loss'
parser.add_argument('--re_loss_option', default='masking', type=str)  # 'none', 'masking', 'selection'

# parser.add_argument('--branches', default='0,0,0,0,0,1', type=str)

parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--alpha_schedule', default=0.50, type=float)

parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--resume', default=False, type=bool)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    log_dir = create_directory(f'./experiments/logs/')
    data_dir = create_directory(f'./experiments/data/')
    model_dir = create_directory('./experiments/models/')
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.tag}/')

    log_path = log_dir + f'{args.tag}.txt'
    data_path = data_dir + f'{args.tag}.json'
    model_path = model_dir + f'{args.tag}.pth'

    set_seed(args.seed)
    log_func = lambda string='': log_print(string, log_path)

    log_func('[i] {}'.format(args.tag))
    log_func()

    # set distributed data parallel

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )
    device = torch.device(f'cuda:{args.local_rank}')

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)

    train_transforms = [
        RandomResize(args.min_image_size, args.max_image_size),
        RandomHorizontalFlip(),
    ]
    if 'colorjitter' in args.augment:
        train_transforms.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))

    if 'randaugment' in args.augment:
        train_transforms.append(RandAugmentMC(n=2, m=10))

    train_transform = transforms.Compose(train_transforms + \
                                         [
                                             Normalize(imagenet_mean, imagenet_std),
                                             RandomCrop(args.image_size),
                                             Transpose()
                                         ]
                                         )
    val_transform = transforms.Compose([
        RandomResize(args.min_image_size, args.max_image_size),
        Normalize(imagenet_mean, imagenet_std),
        RandomCrop(args.image_size),
        Transpose()
    ])

    meta_dic = read_json('./data/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])

    train_dataset = VOC_Dataset_For_Classification(args.data_dir, 'train_aug', train_transform)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              drop_last=True,
                              sampler=train_sampler)

    val_dataset = VOC_Dataset_For_Classification(args.data_dir, 'val', train_transform)
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=1,
                            drop_last=True,
                            sampler=val_sampler)

    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] The number of class is {}'.format(meta_dic['classes']))
    log_func('[i] train_transform is {}'.format(train_transform))
    log_func('[i] test_transform is {}'.format(val_transform))
    log_func()

    #
    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration

    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))

    ###################################################################################
    # Network
    ###################################################################################
    model = vit_base_patch16_224(pretrained=True)

    model = model.cuda()
    model.train()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM' % (calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    class_loss_fn = nn.MultiLabelSoftMarginLoss(reduction='none').cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.wd)

    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train': [],
        'validation': []
    }

    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(['loss'])

    def evaluate(loader):
        all_preds, all_label = [], []
        model.eval()
        eval_timer.tik()

        with torch.no_grad():
            length = len(loader)
            for step, (images, labels) in enumerate(loader):
                images = images.cuda()
                labels = labels.cuda()

                pred_logits = model(images)
                pred_cls = torch.argmax(pred_logits, dim=-1)
                gt_cls = torch.argmax(labels, dim=-1)

                if len(all_preds) == 0:
                    all_preds.append(pred_cls.detach().cpu().numpy())
                    all_label.append(gt_cls.detach().cpu().numpy())

                else:
                    all_preds[0] = np.append(
                        all_preds[0], pred_cls.detach().cpu().numpy(), axis=0
                    )
                    all_label[0] = np.append(
                        all_label[0], gt_cls.detach().cpu().numpy(), axis=0
                    )

        all_preds, all_label = all_preds[0], all_label[0]
        acc = (all_preds == all_label).mean()
        return acc * 100

    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)

    begin_iter = 0
    best_acc = -1
    if args.resume:
        checkpoint_dict = torch.load(model_path)
        model.module.load_state_dict(checkpoint_dict['state_dict'])
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        begin_iter = checkpoint_dict['iter']
        best_acc = checkpoint_dict['best_acc']
        log_func('[i] Loaded checkpoint from {}'.format(model_path))


    ###################################################
    # Begin training
    ###################################################

    for iteration in tqdm(range(begin_iter, max_iteration), desc='Training iter'):

        images, labels = train_iterator.get()
        images, labels = images.cuda(), labels.cuda()

        ###############################################################################
        # Normal
        ###############################################################################
        logits = model(images)

        ###############################################################################
        # Losses
        ###############################################################################

        loss = class_loss_fn(logits, labels).mean()
        reduced_loss = reduce_tensor(loss)

        #################################################################################################

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.local_rank == 0:
            #  reduce losses
            train_meter.add({'loss': reduced_loss.item()})

        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            if args.local_rank == 0:
                loss = train_meter.get(clear=True)

                learning_rate = float(get_learning_rate_from_optimizer(optimizer))

                data = {
                    'iteration': iteration + 1,
                    'learning_rate': learning_rate,
                    'loss': loss,
                    'time': train_timer.tok(clear=True),
                }
                data_dic['train'].append(data)
                write_json(data_path, data_dic)

                log_func('[i] \
                    iteration={iteration:,}, \
                    learning_rate={learning_rate:.4f}, \
                    loss={loss:.4f}, \
                    time={time:.0f}sec'.format(**data)
                         )

                writer.add_scalar('Train/loss', loss, iteration)
                writer.add_scalar('Train/learning_rate', learning_rate, iteration)


        #################################################################################################
        # Evaluation
        #################################################################################################
        if (iteration + 1) % val_iteration == 0:
        # if (iteration + 1) % 10 == 0:
            acc = evaluate(val_loader)
            if args.local_rank == 0:
                if best_acc == -1 or best_acc < acc:
                    best_acc = acc
                    torch.save({'iter': iter,
                                'state_dict': model.module.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'best_acc': best_acc}, model_path)

                    log_func('[i] save model')

                data = {
                    'iteration': iteration + 1,
                    'train_acc': acc,
                    'best_acc': best_acc,
                    'time': eval_timer.tok(clear=True),
                }
                data_dic['validation'].append(data)
                write_json(data_path, data_dic)

                log_func('[i] \
                    iteration={iteration:,}, \
                    train_acc={train_acc:.2f}%, \
                    best_acc={best_acc:.2f}%, \
                    time={time:.0f}sec'.format(**data)
                         )

                writer.add_scalar('Evaluation/acc', acc, iteration)

    write_json(data_path, data_dic)
    writer.close()

    print(args.tag)