import datetime
import torch
import torch.nn as nn

from opencood.tools import train_utils

from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion
from opencood.models.fuse_modules.where2comm_fuse import Where2comm
from opencood.models.fuse_modules.swap_fusion_modules import SwapFusionEncoder
from opencood.models.fuse_modules.v2xvit_basic import V2XTransformer

from opencood.models.sub_modules.resizer import LearnableResizer
from opencood.models.fuse_modules.wg_fusion_modules import CrossDomainFusionEncoder
from opencood.models.da_modules.classfier import DAImgHead

from opencood.models.fuse_modules.fuse_utils import regroup
from einops import repeat
    
class cls_head(nn.Module):
    def __init__(self,args) -> None:
        super(cls_head,self).__init__()
        self.cls_head=nn.Conv2d(
            args['channel'], args["anchor_number"], kernel_size=1
        )
    
    def forward(self,x):
        return self.cls_head(x)
    
class reg_head(nn.Module):
    def __init__(self,args) -> None:
        super(reg_head,self).__init__()
        self.reg_head=nn.Conv2d(args['channel'], 7 * args["anchor_number"], kernel_size=1)
    
    def forward(self,x):
        return self.reg_head(x)

class da(nn.Module):
    def __init__(self, args):
        super(da, self).__init__()
        
        self.fusion_method="fcooper"
        if 'fusion_net' in args and 'core_method' in args['fusion_net']:
            self.fusion_method=args['fusion_net']['core_method']
        assert self.fusion_method=='fcooper' or self.fusion_method=='where2comm' or self.fusion_method=='cobevt' or self.fusion_method=='v2xvit'
        print(f"use {self.fusion_method} fusion method")

        self.encoder_q = train_utils.create_encoder(args["encoder_q"])
        self.encoder_k = train_utils.create_encoder(args["encoder_k"])
        
        self.resizer = LearnableResizer(args['resizer'])
        self.cdt = CrossDomainFusionEncoder(args['cdt'])
        self.classifier = DAImgHead(256)
        
        if self.fusion_method=='fcooper':
            self.fusion_net = SpatialFusion()
        if self.fusion_method=='cobevt':
            self.fusion_net = SwapFusionEncoder(args['fusion_net'])
        if self.fusion_method=='where2comm':
            self.fusion_net = Where2comm(args['fusion_net'])
        if self.fusion_method=='v2xvit':
            self.fusion_net = V2XTransformer(args['fusion_net'])

        self.cls_head = cls_head(args["encoder_q"]['args'])
        self.reg_head = reg_head(args["encoder_q"]['args'])

        self.encoder_q = train_utils.load_pretrained_model(
            args["encoder_q"]["saved_pth"], self.encoder_q
        )
        self.encoder_k = train_utils.load_pretrained_model(
            args["encoder_k"]["saved_pth"], self.encoder_k
        )
        self.cls_head = train_utils.load_pretrained_model(
            args["encoder_q"]["saved_pth"], self.cls_head
        )
        self.reg_head = train_utils.load_pretrained_model(
            args["encoder_q"]["saved_pth"], self.reg_head
        )

        for param_q, param_k, param_cls, param_reg in zip(
            self.encoder_q.parameters(),
            self.encoder_k.parameters(),
            self.cls_head.parameters(),
            self.reg_head.parameters(),
        ):
            param_q.requires_grad = False
            param_k.requires_grad = False
            param_cls.requires_grad = False
            param_reg.requires_grad = False

    def forward(self, data_dict):
        record_len = data_dict["record_len"]
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']
        # B, max_cav, 3(dt dv infra), 1, 1
        prior_encoding =\
            data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)

        with torch.no_grad():
            batch_dict_q = self.encoder_q(data_dict["processed_lidar_q"])
            batch_dict_k = self.encoder_k(data_dict["processed_lidar_k"])

        ego_feature_batch = batch_dict_q["spatial_features_2d"]
        cav_feature_batch = batch_dict_k["spatial_features_2d"]
        reshaped_spatial_feature_list = []
        for i in range(ego_feature_batch.shape[0]):
            ego_feature=ego_feature_batch[i]
            cav_feature=cav_feature_batch[i].unsqueeze(0)
            cav_feature = self.resizer(ego_feature, cav_feature)        
            
            ego_copy = ego_feature[None].repeat(cav_feature.shape[0],
                                                    1, 1, 1)
            cav_feature = self.cdt(ego_copy, cav_feature)

            reshaped_spatial_feature_list.append(torch.cat((ego_feature[None],
                                                               cav_feature),
                                                               dim=0))
            
        spatial_features_2d = torch.cat(reshaped_spatial_feature_list, dim=0)
            
        spatial_features_2d = torch.cat(reshaped_spatial_feature_list, dim=0)
        # for domain classifier
        da_feature = self.classifier(spatial_features_2d)
        
        if self.fusion_method=='cobevt':
            # # N, C, H, W -> B,  L, C, H, W
            regroup_feature, mask = regroup(spatial_features_2d,
                                            record_len,
                                            max_len=5)
            com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            com_mask = repeat(com_mask,
                            'b h w c l -> b (h new_h) (w new_w) c l',
                            new_h=regroup_feature.shape[3],
                            new_w=regroup_feature.shape[4])
            out = self.fusion_net(regroup_feature, com_mask)
        
        if self.fusion_method=='fcooper':
            out = self.fusion_net(spatial_features_2d, record_len)    
        
        if self.fusion_method=='where2comm':
            psm_single = self.cls_head(spatial_features_2d)
            out, communication_rates = self.fusion_net(spatial_features_2d,
                                                                    psm_single,
                                                                    record_len,
                                                                    pairwise_t_matrix)
            
        if self.fusion_method=='v2xvit':
            # N, C, H, W -> B,  L, C, H, W
            regroup_feature, mask = regroup(spatial_features_2d,
                                            record_len,
                                            max_len=5)
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

        psm = self.cls_head(out)
        rm = self.reg_head(out)

        output_dict = {"psm": psm, 
                       "rm": rm,
                       'da_feature': da_feature,
                       'record_len': record_len}

        return output_dict
