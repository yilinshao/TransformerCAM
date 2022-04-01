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
from utils.minisom import MiniSom
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
parser.add_argument('--experiment_name', default='', type=str)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--threshold', default=0.25, type=float)
parser.add_argument('--crf_iteration', default=10, type=int)
parser.add_argument('--from_grad', default=True, type=bool)
parser.add_argument('--use_som', default=False, type=bool)


def get_cam_with_doubt(cams):
    # 得到每一类的threshold
    thresholds_for_classes = []
    for i in range(0, cams.shape[-1]):
        class_cam = cams[..., i]
        # plt.imshow(class_cam)
        # plt.show()

        cam_grad = Sobel(class_cam, cv2.CV_64F, 1, 1, ksize=3)

        # plt.imshow(cam_grad)
        # plt.show()
        max_grad = np.max(cam_grad)
        max_grad_idx = np.where(cam_grad == max_grad)
        thresholds_for_classes.append(class_cam[max_grad_idx[0][0], max_grad_idx[1][0]])

    # 得到预测结果
    pred_mask = np.zeros_like(cams[:, :, 0])
    pred_mask = np.expand_dims(pred_mask, axis=-1)
    for i, class_th in enumerate(thresholds_for_classes):
        pred_class_cam = np.ones_like(cams[:, :, 0]) * -1
        class_cam = cams[..., i]
        class_bg = np.ones_like(cams[:, :, 0]) * class_th
        pred_class_masks = np.argmax(np.concatenate([class_bg[..., np.newaxis], class_cam[..., np.newaxis]], axis=-1),
                                     axis=-1)
        pred_class_cam[pred_class_masks == 1] = class_cam[pred_class_masks == 1]
        pred_mask = np.concatenate([pred_mask, pred_class_cam[..., np.newaxis]], axis=-1)

    pred_mask = np.argmax(pred_mask, axis=-1)

    # 得到每一类背景的threshold
    thresholds_for_class_bg = []
    for th in thresholds_for_classes:
        if th - 0.05 > 0:
            bg_th = th - 0.05
        else:
            bg_th = th
        thresholds_for_class_bg.append(bg_th)

    # 计算每一类的预测图（背景确信）
    pred_bg_mask = np.zeros_like(cams[:, :, 0])
    pred_bg_mask = np.expand_dims(pred_bg_mask, axis=-1)

    for i, bg_th in enumerate(thresholds_for_class_bg):
        pred_bg_cam = np.ones_like(cams[:, :, 0]) * -1
        class_cam = cams[..., i]
        class_bg = np.ones_like(cams[:, :, 0]) * bg_th
        pred_class_masks = np.argmax(np.concatenate([class_bg[..., np.newaxis], class_cam[..., np.newaxis]], axis=-1),
                                     axis=-1)
        pred_bg_cam[pred_class_masks == 1] = class_cam[pred_class_masks == 1]
        pred_bg_mask = np.concatenate([pred_bg_mask, pred_bg_cam[..., np.newaxis]], axis=-1)

    pred_bg_mask = np.argmax(pred_bg_mask, axis=-1)

    conf = pred_mask.copy()
    conf[pred_mask == 0] = 255
    conf[pred_bg_mask + pred_mask == 0] = 0
    # plt.imshow(conf)
    # plt.show()
    return conf


def get_initial_pixels(stacked_features, nlabels):
    initial_pixels_idx = []
    cams = stacked_features[..., -nlabels + 1:]

    # 得到背景的最小值
    fused_cams = np.sum(cams, axis=-1)
    min_value = np.min(fused_cams)
    min_idx = np.where(fused_cams == min_value)
    min_idx = (min_idx[0][0], min_idx[1][0])
    initial_pixels_idx.append(min_idx)

    # 得到每一类的最大值
    for i in range(0, nlabels - 1):
        max_value = np.max(cams[..., i])
        max_idx = np.where(cams[..., i] == max_value)
        max_idx = (max_idx[0][0], max_idx[1][0])
        initial_pixels_idx.append(max_idx)

    initial_pixels = []
    for idx in initial_pixels_idx:
        pixel_vector = stacked_features[idx]
        pixel_vector = np.expand_dims(pixel_vector, axis=0)
        initial_pixels.append(pixel_vector)
    initial_pixels = np.concatenate(initial_pixels, axis=0)

    assert initial_pixels.shape[0] == nlabels

    return initial_pixels


def som_inference_label(ori_image, ori_cams, cams, n_labels):
    # 计算得到包含不确定点(255)的conf
    norm_cam = []
    for ori_cam in ori_cams.transpose((2, 0, 1)):
        norm_cam.append((ori_cam-np.min(ori_cam))/(np.max(ori_cam)-np.min(ori_cam)))
    norm_cam = np.stack(norm_cam, axis=-1)
    ori_cams = norm_cam
    conf = get_cam_with_doubt(ori_cams)

    # 计算每一类的平均像素值
    class_avg_pixels = []

    for label in range(n_labels):
        all_pixels_idx = conf == label
        # cnt = len(np.where(all_pixels_idx == True)[0])
        all_pixels = ori_image[all_pixels_idx]
        avg_pixels = np.mean(all_pixels, axis=0)
        class_avg_pixels.append(avg_pixels)

    # 给每一类赋予像素平均，作为som的输入
    som_input = np.zeros_like(ori_image)
    for label in range(n_labels):
        all_pixels_idx = conf == label
        som_input[all_pixels_idx] = class_avg_pixels[label]

    all_doubt_pixels_idx = conf == 255
    som_input[all_doubt_pixels_idx] = ori_image[all_doubt_pixels_idx]
    # plt.imshow(som_input)
    # plt.show()

    som_input = som_input / 255.
    # cams = cams / np.max(cams)
    # som = MiniSom(x=1, y=n_labels, input_len=ori_image.shape[-1] + cams.shape[-1] + ori_cams.shape[-1], sigma=1, learning_rate=0.2, neighborhood_function='bubble')
    # stacked_features = np.concatenate((cams, ori_image, ori_cams), axis=-1)

    som = MiniSom(x=1, y=n_labels, sigma=0.5, learning_rate=0.2, input_len=ori_image.shape[-1], neighborhood_function='gaussian')
    # stacked_features = np.concatenate((ori_image, ori_cams), axis=-1)
    # init_pixels = get_initial_pixels(stacked_features, n_labels)

    init_pixels = np.stack(class_avg_pixels)
    init_pixels = init_pixels / 255.

    som_input = np.reshape(som_input, (som_input.shape[0]*som_input.shape[1], -1))
    som.fixed_weights_init(init_pixels)
    starting_weights = som.get_weights().copy()

    som.train(som_input, 1000, random_order=True, verbose=False)
    qnt = som.quantization(som_input)
    clustered = np.reshape(qnt, (ori_image.shape[0], ori_image.shape[1], -1))

    # plt.imshow(clustered)
    # plt.show()

    wined_vectors = som.get_weights()

    palette = {}
    for i, init_pixel in enumerate(wined_vectors[0]):
        palette[i] = init_pixel

    mask_label = np.ones((clustered.shape[0], clustered.shape[1]), dtype=np.uint8) * -1
    for c, i in palette.items():
        m = np.all(clustered == np.array(i).reshape(1, 1, clustered.shape[-1]), axis=2)
        mask_label[m] = c

    if np.min(mask_label) == -1:
        raise ValueError(np.where(mask_label == -1))
    # plt.imshow(mask_label)
    # plt.show()

    conf[conf==255] = mask_label[conf == 255]
    return conf


def get_threshold(cams, mask, keys, from_mask, from_grad, from_mean):
    best_mIoU = -1
    cams = cams.transpose((1, 2, 0))
    best_th = None
    if from_mask:
        print('Use mask to generate labels')
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
        return best_th

    elif from_grad:
        print('Use gradient to generate labels')

        # 得到每一类的threshold
        thresholds_for_classes = []
        for i in range(0, cams.shape[-1]):
            class_cam = cams[..., i]
            # plt.imshow(class_cam)
            # plt.show()

            cam_grad = Sobel(class_cam, cv2.CV_64F, 1, 1, ksize=3)
            cam_grad = np.abs(cam_grad)

            # plt.imshow(cam_grad)
            # plt.show()
            max_grad = np.max(cam_grad)
            max_grad_idx = np.where(cam_grad == max_grad)

            min_grad = np.min(cam_grad)
            min_grad_idx = np.where(cam_grad == min_grad)

            max_grad_cam = class_cam[max_grad_idx[0][0], max_grad_idx[1][0]]
            min_grad_cam = class_cam[min_grad_idx[0][0], min_grad_idx[1][0]]
            avg_cam = (max_grad_cam + min_grad_cam + 0.2 ) / 3

            thresholds_for_classes.append(avg_cam)

        # 得到预测结果
        pred_mask = np.zeros_like(cams[:, :, 0])
        pred_mask = np.expand_dims(pred_mask, axis=-1)
        for i, class_th in enumerate(thresholds_for_classes):
            pred_class_cam = np.ones_like(cams[:, :, 0]) * -1
            class_cam = cams[..., i]
            class_bg = np.ones_like(cams[:, :, 0]) * class_th
            pred_class_masks = np.argmax(np.concatenate([class_bg[..., np.newaxis], class_cam[..., np.newaxis]], axis=-1), axis=-1)
            pred_class_cam[pred_class_masks == 1] = class_cam[pred_class_masks == 1]
            pred_mask = np.concatenate([pred_mask, pred_class_cam[..., np.newaxis]], axis=-1)

        pred_mask = np.argmax(pred_mask, axis=-1)
        # plt.imshow(pred_mask)
        # plt.show()

        # 计算mIoU
        # pred_class_key = np.copy(pred_mask)
        # for i, key in enumerate(keys):
        #     pred_class_key[pred_class_key == i] = key
        # meter_dic = {'grad_th': Calculator_For_mIoU('./data/VOC_2012.json')}
        # meter_dic['grad_th'].add(pred_class_key, mask)
        # mIoU, mIoU_foreground = meter_dic['grad_th'].get(clear=True)
        # print(mIoU)

        return pred_mask

    elif from_mean:
        print('Use mean value to generate labels')

        thresholds = [np.mean(cams)]
        meter_dic = {th: Calculator_For_mIoU('./data/VOC_2012.json') for th in thresholds}

        for th in thresholds:
            bg = np.ones_like(cams[:, :, 0]) * th
            pred_masks = np.argmax(np.concatenate([bg[..., np.newaxis], cams], axis=-1), axis=-1)

            assert keys.shape[0] == cams.shape[-1] + 1
            for i, key in enumerate(keys):
                pred_masks[pred_masks == i] = key

            meter_dic[th].add(pred_masks, mask)
        #
        # for th in thresholds:
        #     mIoU, mIoU_foreground = meter_dic[th].get(clear=True)
        #     if best_mIoU < mIoU:
        #         best_th = th
        #         best_mIoU = mIoU
        #
        # print(best_mIoU)
        return thresholds[0]

    else:
        raise ValueError("Unsupported mode")

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    cam_dir = f'./experiments/predictions/{args.experiment_name}/'
    pred_dir = create_directory(f'./experiments/predictions/{args.experiment_name}@crf={args.crf_iteration}@grad/')

    set_seed(args.seed)
    log_func = lambda string='': print(string)
    
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()
    matrix = Calculator_For_mIoU('./data/VOC_2012.json')

    with torch.no_grad():
        length = len(dataset)
        for step, (ori_image, image_id, label, gt_mask) in enumerate(dataset):
            png_path = pred_dir + image_id + '.png'
            # if os.path.isfile(png_path):
            #     continue
            
            ori_w, ori_h = ori_image.size
            predict_dict = np.load(cam_dir + image_id + '.npy', allow_pickle=True).item()
            gt_mask = np.asarray(gt_mask)
            keys = predict_dict['keys']
            
            cams = predict_dict['rw']
            ori_cams = cams.copy()

            if args.from_grad == True:
                cams = get_threshold(cams, gt_mask, keys, from_grad=True, from_mask=False, from_mean=False)

            elif args.use_som == True:
                cams = som_inference_label(np.asarray(ori_image), ori_cams.transpose((1, 2, 0)), cams[..., np.newaxis],
                                           n_labels=keys.shape[0])

            else:
                args.threshold = get_threshold(cams, gt_mask, keys, from_grad=False, from_mask=False, from_mean=True)
                cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)

                cams = np.argmax(cams, axis=0)

            # cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)
            #
            # cams = np.argmax(cams, axis=0)

            if args.crf_iteration > 0:
                cams = crf_inference_label(np.asarray(ori_image), cams, n_labels=keys.shape[0], t=args.crf_iteration)
            conf = keys[cams]
            imageio.imwrite(png_path, conf.astype(np.uint8))


            matrix.add(conf, gt_mask)
            mIoU, mIoU_foreground, IoU_dic, TP, TN, FP, FN = matrix.get(detail=True, clear=False)
            print('mIoU:{}, mIoU_foreground:{}, {}, TP:{}, TN:{}, FP:{}, FN:{}'.format(mIoU,
                                                                                       mIoU_foreground,
                                                                                       IoU_dic, TP, TN, FP, FN))

            # cv2.imshow('image', np.asarray(ori_image))
            # cv2.imshow('predict', decode_from_colormap(conf, dataset.colors))
            # cv2.waitKey(0)

            # plt.imshow(np.asarray(ori_image))
            # plt.show()
            # plt.imshow(decode_from_colormap(conf, dataset.colors))
            # plt.show()
            # plt.imshow(decode_from_colormap(gt_mask.astype(np.int32), dataset.colors))
            # plt.show()

            sys.stdout.write('\r# Make Pseudo Labels [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), conf.shape))
            sys.stdout.flush()
        print()
    
    print("python3 evaluate.py --experiment_name {} --mode png".format(args.experiment_name + f'@crf={args.crf_iteration}'))
