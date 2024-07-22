from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.tools import train_utils
from opencood.models.sub_modules.compressor import simple_align

# from opencood.models.fuse_modules.maxout import Maxout
from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion
from opencood.models.fuse_modules.where2comm_fuse import Where2comm
from opencood.models.fuse_modules.swap_fusion_modules import SwapFusionEncoder
from opencood.models.fuse_modules.v2xvit_basic import V2XTransformer
from opencood.models.fuse_modules.fuse_utils import regroup

    
class detect_head(nn.Module):
    def __init__(self, args) -> None:
        super(detect_head, self).__init__()
        in_channel=args['channel']
        self.cls_head = nn.Conv2d(in_channel, args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(
            in_channel, 7 * args["anchor_number"], kernel_size=1
        )

    def forward(self, x):
        psm = self.cls_head(x)
        rm = self.reg_head(x)
        return {"psm": psm, "rm": rm}

class hete(nn.Module):
    def __init__(self, args):
        super(hete, self).__init__()

        self.fusion_method="fcooper"
        if 'fusion_net' in args and 'core_method' in args['fusion_net']:
            self.fusion_method=args['fusion_net']['core_method']
        assert self.fusion_method=='fcooper' or self.fusion_method=='where2comm' or self.fusion_method=='cobevt' or self.fusion_method=='v2xvit'
        print(f"use {self.fusion_method} fusion method")

        
        self.encoder_q = train_utils.create_encoder(args["encoder_q"])
        self.encoder_k = train_utils.create_encoder(args["encoder_k"])
        
        if self.fusion_method=='fcooper':
            self.fusion_net = SpatialFusion()
        if self.fusion_method=='cobevt':
            self.fusion_net = SwapFusionEncoder(args['fusion_net'])
        if self.fusion_method=='where2comm':
            self.fusion_net = Where2comm(args['fusion_net'])
        if self.fusion_method=='v2xvit':
            self.fusion_net = V2XTransformer(args['fusion_net'])
            
        self.simple_align = simple_align(args['compressor'])

        self.detect_head = detect_head(args["encoder_q"]['args'])

        self.encoder_q = train_utils.load_pretrained_model(
            args["encoder_q"]["saved_pth"], self.encoder_q
        )
        self.encoder_k = train_utils.load_pretrained_model(
            args["encoder_k"]["saved_pth"], self.encoder_k
        )
        self.detect_head = train_utils.load_pretrained_model(
            args["encoder_q"]["saved_pth"], self.detect_head
        )
        
        for param_q, param_k,param_h in zip(
            self.encoder_q.parameters(),
            self.encoder_k.parameters(),
            self.detect_head.parameters(),
        ):
            param_q.requires_grad = False
            param_k.requires_grad = False
            param_h.requires_grad = False

    def forward(self, data_dict):
        record_len = data_dict["record_len"]
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        spatial_correction_matrix = data_dict['spatial_correction_matrix'][:,:2,:,:]
        # B, max_cav, 3(dt dv infra), 1, 1
        prior_encoding =\
            data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)[:,:2,:,:,:]

        with torch.no_grad():
            batch_dict_q = self.encoder_q(data_dict["processed_lidar_q"])
            batch_dict_k = self.encoder_k(data_dict["processed_lidar_k"])

        query = batch_dict_q["spatial_features_2d"]
        key = batch_dict_k["spatial_features_2d"]
        key = self.simple_align(key)
        
        spatial_features_2d=[]
        for i in range(query.shape[0]):
            spatial_features_2d.append(query[i])
            spatial_features_2d.append(key[i])
        spatial_features_2d = torch.stack(spatial_features_2d,dim=0)
            
        if self.fusion_method=='cobevt':
            # # N, C, H, W -> B,  L, C, H, W
            regroup_feature, mask = regroup(spatial_features_2d,
                                            record_len,
                                            max_len=2)
            com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            com_mask = repeat(com_mask,
                            'b h w c l -> b (h new_h) (w new_w) c l',
                            new_h=regroup_feature.shape[3],
                            new_w=regroup_feature.shape[4])
            out = self.fusion_net(regroup_feature, com_mask)
        
        if self.fusion_method=='fcooper':
            out = self.fusion_net(spatial_features_2d, record_len)
        
        if self.fusion_method=='where2comm':
            psm_single = self.detect_head.cls_head(spatial_features_2d)
            out, communication_rates = self.fusion_net(spatial_features_2d,
                                                                    psm_single,
                                                                    record_len,
                                                                    pairwise_t_matrix)
            
        if self.fusion_method=='v2xvit':
            # N, C, H, W -> B,  L, C, H, W
            regroup_feature, mask = regroup(spatial_features_2d,
                                            record_len,
                                            max_len=2)
            # prior encoding added
            prior_encoding = prior_encoding.repeat(1, 1, 1,
                                                regroup_feature.shape[3],
                                                regroup_feature.shape[4])
            regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)

            # b l c h w -> b l h w c
            regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
            # transformer fusion
            fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
            # b h w c -> b c h w
            out = fused_feature.permute(0, 3, 1, 2)

        output_dict = self.detect_head(out)

        return output_dict
