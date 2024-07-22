# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.att_bev_backbone  import AttBEVBackbone

from opencood.hypes_yaml.yaml_utils import save_yaml
from opencood.utils.common_utils import torch_tensor_to_numpy


class SecondIntermediate(nn.Module):
    def __init__(self, args):
        super(SecondIntermediate, self).__init__()
        self.save_feature_flag = False
        if 'save_features' in args:
            self.save_feature_flag = args['save_features']
            self.save_folder = args['output_folder']
        print('save_features:',self.save_feature_flag)

        self.batch_size = args['batch_size']
        self.batch_size = 1 
        # mean_vfe
        self.mean_vfe = MeanVFE(args['mean_vfe'], 4)
        # sparse 3d backbone
        self.backbone_3d = VoxelBackBone8x(args['backbone_3d'],
                                           4, args['grid_size'])
        # height compression
        self.height_compression = HeightCompression(args['height_compression'])
        # base ben backbone
        self.backbone_2d = AttBEVBackbone(args['base_bev_backbone'], 256)

        # head
        self.cls_head = nn.Conv2d(256 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(256 * 2, 7 * args['anchor_num'],
                                  kernel_size=1)

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'batch_size': torch.sum(record_len).cpu().numpy(),
                      'record_len': record_len}

        batch_dict = self.mean_vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.height_compression(batch_dict)
        if self.save_feature_flag:
            self.save_feature(batch_dict['spatial_features'], data_dict)
        batch_dict = self.backbone_2d(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict
    
    def save_feature(self, spatial_2d, data_dict):
        """
        Save the features in the folder for later training.

        Parameters
        ----------
        spatial_2d : torch.tensor
            Spatial features, N C H W

        data_dict: dict
            Metadata.
        """
        index = 0

        for cav_id, cav_content in data_dict['raw_info'][0].items():
            scene = cav_content['scene']
            output_folder = os.path.join(self.save_folder, scene, cav_id)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            timestamp = cav_content['timestamp']

            if not cav_content['valid']:
                save_array = \
                    np.zeros_like(torch_tensor_to_numpy(spatial_2d[0]))
            else:
                save_array = torch_tensor_to_numpy(spatial_2d[index])
                index += 1

            # save the data
            save_yml_name = os.path.join(output_folder, timestamp + '.yaml')
            save_feature_name = os.path.join(output_folder, timestamp + '.npz')
            save_yaml(cav_content['yaml'], save_yml_name)
            np.savez_compressed(save_feature_name, save_array)