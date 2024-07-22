# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone


class SecondEncoder(nn.Module):
    def __init__(self, args):
        super(SecondEncoder, self).__init__()

        self.batch_size = args['batch_size']
        # self.batch_size = 1
        # mean_vfe
        self.mean_vfe = MeanVFE(args['mean_vfe'], 4)
        # sparse 3d backbone
        self.backbone_3d = VoxelBackBone8x(args['backbone_3d'],
                                           4, args['grid_size'])
        print(args['grid_size'])
        # height compression
        self.height_compression = HeightCompression(args['height_compression'])
        # base ben backbone
        self.backbone_2d = BaseBEVBackbone(args['base_bev_backbone'], 256)

    def forward(self, data_dict):
        data_dict['batch_size'] = self.batch_size

        data_dict = self.mean_vfe(data_dict)
        data_dict = self.backbone_3d(data_dict)
        data_dict = self.height_compression(data_dict)
        data_dict = self.backbone_2d(data_dict)

        return data_dict