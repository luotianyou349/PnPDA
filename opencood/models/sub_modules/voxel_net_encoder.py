import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

from opencood.models.voxel_net import CML
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.utils.common_utils import torch_tensor_to_numpy


class VoxelNetEncoder(nn.Module):
    def __init__(self, args) -> None:
        super(VoxelNetEncoder, self).__init__()
        self.svfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"],
        )
        self.cml = CML()

        self.N = args["N"]
        self.D = args["D"]
        self.H = args["H"]
        self.W = args["W"]
        self.T = args["T"]

        self.N = 1
        print(self.N, self.D, self.H, self.W)

    def voxel_indexing(self, sparse_features, coords):
        dim = sparse_features.shape[-1]

        dense_feature = Variable(
            torch.zeros(dim, self.N, self.D, self.H, self.W).cuda()
        )
        # print('dense_feature',dense_feature.shape)

        dense_feature[
            :, coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        ] = sparse_features.transpose(0, 1)

        return dense_feature.transpose(0, 1)

    def forward(self, data_dict,record_len_tmp=None):
        if record_len_tmp is not None:
            # record_len_tmp = list(record_len_tmp.numpy())
            self.N = sum(record_len_tmp)
            # print(self.N)

        # print('pillar_features in VoxelNetEncoder',data_dict['voxel_features'].shape)
        # feature learning network
        vwfs = self.svfe(data_dict)["pillar_features"]
        # print('vwfs in VoxelNetEncoder',vwfs.shape)

        voxel_coords = torch_tensor_to_numpy(data_dict["voxel_coords"])
        vwfs = self.voxel_indexing(vwfs, voxel_coords)
        # print('vwfs in VoxelNetEncoder',vwfs.shape)

        # convolutional middle network
        vwfs = self.cml(vwfs)
        # convert from 3d to 2d N C H W
        vmfs = vwfs.view(self.N, -1, self.H, self.W)
        data_dict["spatial_features_2d"] = vmfs

        # print('vmfs in VoxelNetEncoder',vmfs.shape)
        # exit(0)

        return data_dict
