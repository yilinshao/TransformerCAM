# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

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

from cv2 import Sobel

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--experiment_name', default='resnet50@seed=0@bs=16@ep=5@nesterov@train@scale=0.5,1.0,1.5,2.0',
                    type=str)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--fg_threshold', default=0.30, type=float)
parser.add_argument('--bg_threshold', default=0.05, type=float)

parser.add_argument('--calculate_mIoU', default=True, type=bool)


def get_threshold(cams, mask, keys, from_mask, from_mean, calculate_miou):
    best_mIoU = -1
    cams = cams.transpose((1, 2, 0))
    best_th = None
    if from_mask:
        assert mask is not None
        thresholds = list(np.arange(0.10, 0.70, 0.02))
        meter_dic = {th: Calculator_For_mIoU('./data/VOC_2012.json') for th in thresholds}

        for th in thresholds:
            bg = np.ones_like(cams[:, :, 0]) * th
            pred_masks = np.argmax(np.concatenate([bg[..., np.newaxis], cams], axis=-1), axis=-1)

            assert keys.shape[0] == cams.shape[-1] + 1
            for i, key in enumerate(keys):
                pred_masks[pred_masks == i] = key

            meter_dic[th].add(pred_masks, mask)

        for th in thresholds:
            mIoU, mIoU_foreground = meter_dic[th].get(clear=True)
            if best_mIoU < mIoU:
                best_th = th
                best_mIoU = mIoU
        print(best_mIoU)

        ########################
        # ?????????????????????mIoU???
        ########################
        if calculate_miou:
            return best_th, best_th

        ########################
        # ????????????????????????????????????
        ########################
        else:
            return min(best_th + 0.1, 0.9), max(best_th - 0.1, 0.1)


    elif from_mean:

        thresholds = [np.mean(cams)]
        # meter_dic = {th: Calculator_For_mIoU('./data/VOC_2012.json') for th in thresholds}
        #
        # for th in thresholds:
        #     bg = np.ones_like(cams[:, :, 0]) * th
        #     pred_masks = np.argmax(np.concatenate([bg[..., np.newaxis], cams], axis=-1), axis=-1)
        #
        #     assert keys.shape[0] == cams.shape[-1] + 1
        #     for i, key in enumerate(keys):
        #         pred_masks[pred_masks == i] = key
        #
        #     meter_dic[th].add(pred_masks, mask)

        # for th in thresholds:
        #     mIoU, mIoU_foreground = meter_dic[th].get(clear=True)
        #     if best_mIoU < mIoU:
        #         best_th = th
        #         best_mIoU = mIoU
        #
        # print(best_mIoU)

        ########################
        # ?????????????????????mIoU???
        ########################
        if calculate_miou:
            return thresholds[0], thresholds[0]

        ########################
        # ????????????????????????????????????
        ########################
        else:
            return min(thresholds[0] + 0.1, 0.9), max(thresholds[0] - 0.1, 0.1)


    else:
        raise ValueError("Unsupported mode")



if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    experiment_name = args.experiment_name

    pred_dir = f'./experiments/predictions/{experiment_name}/'
    aff_dir = create_directory(
        './experiments/predictions/{}@aff_fg={:.2f}_bg={:.2f}/'.format(experiment_name, args.fg_threshold,
                                                                       args.bg_threshold))

    set_seed(args.seed)
    log_func = lambda string='': print(string)

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    # for mIoU
    meta_dic = read_json('./data/VOC_2012.json')
    dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)

    #################################################################################################
    # Convert
    #################################################################################################
    eval_timer = Timer()
    matrix = Calculator_For_mIoU('./data/VOC_2012.json')
    length = len(dataset)
    for step, (ori_image, image_id, _, mask) in enumerate(dataset):
        png_path = aff_dir + image_id + '.png'
        # if os.path.isfile(png_path):
        #     continue

        # load
        image = np.asarray(ori_image)
        mask = np.asarray(mask)
        # plt.imshow(ori_image)
        # plt.show()
        cam_dict = np.load(pred_dir + image_id + '.npy', allow_pickle=True).item()

        ori_h, ori_w, c = image.shape

        keys = cam_dict['keys']
        cams = cam_dict['hr_cam']

        args.fg_threshold, args.bg_threshold = get_threshold(cams, mask, keys,
                                                             from_mask=True,
                                                             from_mean=False,
                                                             calculate_miou=args.calculate_mIoU)
        # 1. find confident fg & bg
        fg_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.fg_threshold)
        fg_cam = np.argmax(fg_cam, axis=0)
        # plt.imshow(fg_cam)
        # plt.show()
        fg_conf = keys[crf_inference_label(image, fg_cam, n_labels=keys.shape[0])]
        # plt.imshow(fg_conf)
        # plt.show()

        bg_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.bg_threshold)
        bg_cam = np.argmax(bg_cam, axis=0)
        # plt.imshow(bg_cam)
        # plt.show()

        bg_conf = keys[crf_inference_label(image, bg_cam, n_labels=keys.shape[0])]
        # plt.imshow(bg_conf)
        # plt.show()
        # 2. combine confident fg & bg
        conf = fg_conf.copy()
        conf[fg_conf == 0] = 255
        conf[bg_conf + fg_conf == 0] = 0
        # plt.imshow(conf)
        # plt.show()

        ####################################################
        # ????????????????????????mIoU????????????????????????????????????????????????mean
        ####################################################
        if args.calculate_mIoU:
            pred_mask = fg_conf
            matrix.add(conf, mask)
            mIoU, mIoU_foreground, IoU_dic, TP, TN, FP, FN = matrix.get(detail=True, clear=False)
            print('mIoU:{}, mIoU_foreground:{}, {}, TP:{}, TN:{}, FP:{}, FN:{}'.format(mIoU,
                                                                                       mIoU_foreground,
                                                                                       IoU_dic, TP, TN, FP, FN))

        else:
            imageio.imwrite(png_path, conf.astype(np.uint8))

        sys.stdout.write('\r# Convert [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100,
                                                                          (ori_h, ori_w), conf.shape))
        sys.stdout.flush()
    print()
