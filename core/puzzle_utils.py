import math

import torch
import torch.nn.functional as F

def tile_features(features, num_pieces):
    _, _, h, w = features.size()

    num_pieces_per_line = int(math.sqrt(num_pieces))
    
    h_per_patch = h // num_pieces_per_line
    w_per_patch = w // num_pieces_per_line
    
    """
    +-----+-----+
    |  1  |  2  |
    +-----+-----+
    |  3  |  4  |
    +-----+-----+

    +-----+-----+-----+-----+
    |  1  |  2  |  3  |  4  |
    +-----+-----+-----+-----+
    """
    patches = []
    for splitted_features in torch.split(features, h_per_patch, dim=2):
        for patch in torch.split(splitted_features, w_per_patch, dim=3):
            patches.append(patch)
    
    return torch.cat(patches, dim=0)

def merge_features(features, num_pieces, batch_size):
    """
    +-----+-----+-----+-----+
    |  1  |  2  |  3  |  4  |
    +-----+-----+-----+-----+
    
    +-----+-----+
    |  1  |  2  |
    +-----+-----+
    |  3  |  4  |
    +-----+-----+
    """
    features_list = list(torch.split(features, batch_size))
    num_pieces_per_line = int(math.sqrt(num_pieces))
    
    index = 0
    ext_h_list = []

    for _ in range(num_pieces_per_line):

        ext_w_list = []
        for _ in range(num_pieces_per_line):
            ext_w_list.append(features_list[index])
            index += 1
        
        ext_h_list.append(torch.cat(ext_w_list, dim=3))

    features = torch.cat(ext_h_list, dim=2)
    return features


def merge_att_mat(att_mats, num_pieces, batch_size):
    att_mat_list = list(torch.split(att_mats, batch_size))
    att_mat = att_mat_list[0]
    for i in range(len(att_mat_list)):
        att_mat = torch.cat((att_mat, att_mat_list[i][:, 1:]), dim=1)
    return att_mat


def puzzle_module(x, func_list, num_pieces):
    tiled_x = tile_features(x, num_pieces)

    for func in func_list:
        tiled_x = func(tiled_x)
        
    merged_x = merge_features(tiled_x, num_pieces, x.size()[0])
    return merged_x
