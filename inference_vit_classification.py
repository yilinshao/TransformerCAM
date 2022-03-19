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

# from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_LRP import vit_large_patch16_384 as vit_LRP

from baselines.ViT.helpers import *
from baselines.ViT.ViT_explanation_generator import Baselines, LRP

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--mode', default='normal', type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--domain', default='train', type=str)
parser.add_argument('--image_size', required=True, type=int)

parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_trained_vit_with_lrp(model_name):
    model_LRP = vit_LRP(pretrained=False).cuda()

    state_dict_path = 'experiments/models/{}.pth'.format(model_name)
    state_dict = load_state_dict(state_dict_path)
    model_LRP.load_state_dict(state_dict)

    model_LRP.eval()
    lrp = LRP(model_LRP)

    return lrp

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    experiment_name = args.tag

    image_size = args.image_size

    if 'train' in args.domain:
        experiment_name += '@train'
    else:
        experiment_name += '@val'

    experiment_name += '@scale=%s' % args.scales

    pred_dir = create_directory(f'./experiments/predictions/{experiment_name}/')

    model_path = './experiments/models/' + f'{args.tag}.pth'

    set_seed(args.seed)
    log_func = lambda string='': print(string)

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    image_transform = transforms.Compose([
        RandomResize(image_size, image_size),
        Normalize(imagenet_mean, imagenet_std),
        Top_Left_Crop(image_size),
    ])

    # for mIoU
    meta_dic = read_json('./data/VOC_2012.json')
    dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)


    ###################################################################################
    # Network
    ###################################################################################
    model = get_trained_vit_with_lrp(model_name=args.tag)

    # log_func('[i] Architecture is {}'.format(args.architecture))
    # log_func('[i] Total Params: %.2fM' % (calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]

    eval_timer.tik()

    def get_cam(ori_image, index):
        # preprocessing
        image = copy.deepcopy(ori_image)

        embedded_image_size = int(image_size / 16)

        w, h = image.size
        if w < h:
            scale = image_size / h
        else:
            scale = image_size / w

        resized_w = int(round(w * scale))
        resized_h = int(round(h * scale))


        image = image_transform(image)
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        flipped_image = image.flip(-1)


        image.requires_grad = True
        image = image.requires_grad_()

        flipped_image.requires_grad = True
        flipped_image = flipped_image.requires_grad_()

        image_Res = model.generate_LRP(image.cuda(),
                                       index=index,
                                         start_layer=0,
                                         method="transformer_attribution").reshape(1, 1, embedded_image_size, embedded_image_size)
        image_Res = torch.nn.functional.interpolate(image_Res, scale_factor=16, mode='bilinear').cuda()
        ori_image_Res = image_Res[..., : resized_h, : resized_w]
        ori_image_Res = (ori_image_Res - ori_image_Res.min()) / (ori_image_Res.max() - ori_image_Res.min())

        filpped_image_Res = model.generate_LRP(flipped_image.cuda(),
                                               index=index,
                                         start_layer=1,
                                         method="transformer_attribution").reshape(1, 1, embedded_image_size, embedded_image_size)

        filpped_image_Res = torch.nn.functional.interpolate(filpped_image_Res.flip(-1), scale_factor=16, mode='bilinear').cuda()
        ori_flipped_image_Res = filpped_image_Res[..., : resized_h, : resized_w]
        ori_flipped_image_Res = (ori_flipped_image_Res - ori_flipped_image_Res.min()) / (ori_flipped_image_Res.max() - ori_flipped_image_Res.min())

        cams = (ori_image_Res + ori_flipped_image_Res) / 2.

        return cams.squeeze(0)

    show_cam = True
    length = len(dataset)
    for step, (ori_image, image_id, label, gt_mask) in enumerate(dataset):
        ori_w, ori_h = ori_image.size

        npy_path = pred_dir + image_id + '.npy'
        # if os.path.isfile(npy_path):
        #     continue

        strided_size = get_strided_size((ori_h, ori_w), 4)
        strided_up_size = get_strided_up_size((ori_h, ori_w), 16)

        keys = torch.nonzero(torch.from_numpy(label))[:, 0]
        cams_list = [get_cam(ori_image, key) for key in keys]

        strided_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_size)[0] for cams in cams_list]
        strided_cams = torch.cat(strided_cams_list)

        hr_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_up_size)[0] for cams in cams_list]
        hr_cams = torch.cat(hr_cams_list)[:, :ori_h, :ori_w]

        strided_cams /= F.adaptive_max_pool2d(strided_cams, (1, 1)) + 1e-5

        hr_cams /= F.adaptive_max_pool2d(hr_cams, (1, 1)) + 1e-5

        if show_cam:
            orig_size_cams = [resize_for_tensors(cams.unsqueeze(0), (ori_h, ori_w))[0] for cams in cams_list]
            orig_size_img = np.asarray(ori_image).astype(np.uint8)

            for ori_size_cam in orig_size_cams:
                colored_cam = (ori_size_cam[0].detach().cpu().numpy() * 255).astype(np.uint8)

                colored_cam = colormap(colored_cam)
                fused_cam_ori_img = cv2.addWeighted(orig_size_img[..., ::-1], 0.5, colored_cam, 0.5, 0)[..., ::-1]
                fused_cam_ori_img = fused_cam_ori_img.astype(np.float32) / 255.
                plt.imshow(fused_cam_ori_img)
                plt.show()
        # save cams
        keys = np.pad(keys + 1, (1, 0), mode='constant')
        if show_cam is not True:
            np.save(npy_path, {"keys": keys, "cam": strided_cams.detach().cpu().numpy(), "hr_cam": hr_cams.detach().cpu().numpy()})

        sys.stdout.write(
            '\r# Make CAM [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100,
                                                              (ori_h, ori_w), hr_cams.size()))
        sys.stdout.flush()
    print()

    if args.domain == 'train_aug':
        args.domain = 'train'

    print("python3 evaluate.py --experiment_name {} --domain {}".format(experiment_name, args.domain))