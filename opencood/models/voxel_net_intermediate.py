# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
VoxelNet for intermediate fusion
"""
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

from opencood.models.voxel_net import RPN, CML
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.models.fuse_modules.self_attn import AttFusion
from opencood.models.sub_modules.auto_encoder import AutoEncoder

from opencood.hypes_yaml.yaml_utils import save_yaml

# conv2d + bn + relu
class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, activation=True,
                 batch_norm=True, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k,
                              stride=s, padding=p, bias=bias)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


class NaiveFusion(nn.Module):

    def __init__(self):
        super(NaiveFusion, self).__init__()
        self.conv1 = Conv2d(128 * 5, 256, 3, 1, 1,
                            batch_norm=False, bias=False)
        self.conv2 = Conv2d(256, 128, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class VoxelNetIntermediate(nn.Module):
    def __init__(self, args):
        super(VoxelNetIntermediate, self).__init__()
        self.save_feature_flag = False
        if 'save_features' in args:
            self.save_feature_flag = args['save_features']
            self.save_folder = args['output_folder']
        print('save_features:',self.save_feature_flag)
        
        self.svfe = PillarVFE(args['pillar_vfe'],
                              num_point_features=4,
                              voxel_size=args['voxel_size'],
                              point_cloud_range=args['lidar_range'])
        self.cml = CML()
        self.fusion_net = AttFusion(128)
        self.rpn = RPN(args['anchor_num'])

        self.N = args['N']
        self.D = args['D']
        self.H = args['H']
        self.W = args['W']
        self.T = args['T']
        self.N=1
        print(self.N, self.D, self.H, self.W)
        self.anchor_num = args['anchor_num']

        self.compression = False
        if 'compression' in args and args['compression'] > 0:
            self.compression = True
            self.compression_layer = AutoEncoder(128, args['compression'])

    def voxel_indexing(self, sparse_features, coords):
        dim = sparse_features.shape[-1]

        dense_feature = Variable(
            torch.zeros(dim, self.N, self.D, self.H, self.W).cuda())
        # print('dense_feature',dense_feature.shape)

        dense_feature[:, coords[:, 0], coords[:, 1], coords[:, 2],
        coords[:, 3]] = sparse_features.transpose(0, 1)

        return dense_feature.transpose(0, 1)

    def regroup(self, dense_feature, record_len):
        """
        Regroup the data based on the record_len.

        Parameters
        ----------
        dense_feature : torch.Tensor
            N, C, H, W
        record_len : list
            [sample1_len, sample2_len, ...]

        Returns
        -------
        regroup_feature : torch.Tensor
            B, 5C, H, W
        """
        cum_sum_len = list(np.cumsum(record_len))
        split_features = torch.tensor_split(dense_feature,
                                            cum_sum_len[:-1])
        regroup_features = []

        for split_feature in split_features:
            # M, C, H, W
            feature_shape = split_feature.shape

            # the maximum M is 5 as most 5 cavs
            padding_len = 5 - feature_shape[0]
            padding_tensor = torch.zeros(padding_len, feature_shape[1],
                                         feature_shape[2], feature_shape[3])
            padding_tensor = padding_tensor.to(split_feature.device)

            split_feature = torch.cat([split_feature, padding_tensor],
                                      dim=0)

            # 1, 5C, H, W
            split_feature = split_feature.view(-1,
                                               feature_shape[2],
                                               feature_shape[3]).unsqueeze(0)
            regroup_features.append(split_feature)

        # B, 5C, H, W
        regroup_features = torch.cat(regroup_features, dim=0)

        return regroup_features

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        # print(record_len)

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        if voxel_coords.is_cuda:
            record_len_tmp = record_len.cpu()

        record_len_tmp = list(record_len_tmp.numpy())

        self.N = sum(record_len_tmp)
        # print(self.N)

        # feature learning network
        vwfs = self.svfe(batch_dict)['pillar_features']
        # print('vwfs in svfe',vwfs.shape)

        voxel_coords = torch_tensor_to_numpy(voxel_coords)
        # print('voxel_coords',voxel_coords.shape)
        vwfs = self.voxel_indexing(vwfs, voxel_coords)
        # print('vwfs in voxel_indexing',vwfs.shape)

        # convolutional middle network
        vwfs = self.cml(vwfs)
        # print('vwfs in cml',vwfs.shape)
        # convert from 3d to 2d N C H W
        vmfs = vwfs.view(self.N, -1, self.H, self.W)
        # print('vwfs in view',vwfs.shape)
        if self.save_feature_flag:
            self.save_feature(vmfs, data_dict)

        # compression layer
        # if self.compression:
        #     vmfs = self.compression_layer(vmfs)

        # information naive fusion
        vmfs_fusion = self.fusion_net(vmfs, record_len)

        # region proposal network
        # merge the depth and feature dim into one, output probability score
        # map and regression map
        psm, rm = self.rpn(vmfs_fusion)

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