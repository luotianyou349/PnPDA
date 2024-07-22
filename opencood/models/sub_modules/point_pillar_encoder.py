import torch
import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone

class PointPillarEncoder(nn.Module):
    def __init__(self, args):
        super(PointPillarEncoder, self).__init__()

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        
    def forward(self, data_dict):
        
        # n, 4 -> n, c
        # print('voxel_features',data_dict['voxel_features'].shape)
        data_dict = self.pillar_vfe(data_dict)
        # n, c -> N, C, H, W
        # print('pillar_features',data_dict['pillar_features'].shape)
        data_dict = self.scatter(data_dict)
        # print('spatial_features',data_dict['spatial_features'].shape)
        data_dict = self.backbone(data_dict)
        # print('spatial_features_2d',data_dict['spatial_features_2d'].shape)

        return data_dict