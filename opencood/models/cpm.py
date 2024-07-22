import datetime
import os
from einops import repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.tools import train_utils
from opencood.models.sub_modules.compressor import simple_align,simple_align_v2
from opencood.models.sub_modules.adapter import TransformerDecoder,TransformerEncoder

from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion
from opencood.models.fuse_modules.where2comm_fuse import Where2comm
from opencood.models.fuse_modules.swap_fusion_modules import SwapFusionEncoder
from opencood.models.fuse_modules.v2xvit_basic import V2XTransformer
from opencood.models.fuse_modules.fuse_utils import regroup
from opencood.visualization.show import feature_show

from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.hypes_yaml.yaml_utils import save_yaml

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
    

class cpm(nn.Module):
    def __init__(self, args):
        super(cpm, self).__init__()
            
        self.momentum = args["momentum"]
        
        self.USE_PRED=args['USE_PRED']
        print("USE_PRED:",self.USE_PRED)
        self.PROJ_QUERY=args['PROJ_QUERY']
        print("PROJ_QUERY:",self.PROJ_QUERY)
        
        self.fusion_method="fcooper"
        if 'fusion_net' in args and 'core_method' in args['fusion_net']:
            self.fusion_method=args['fusion_net']['core_method']
        assert self.fusion_method=='fcooper' or self.fusion_method=='where2comm' or self.fusion_method=='cobevt' or self.fusion_method=='v2xvit'
        print(f"USE {self.fusion_method} fusion method")

        self.encoder_q = train_utils.create_encoder(args["encoder_q"])
        self.encoder_k = train_utils.create_encoder(args["encoder_k"])
        self.compressor = simple_align(args['compressor'])
        self.proj_q = TransformerEncoder(args["projector"])
        self.proj_k = TransformerEncoder(args["projector"])
        if self.USE_PRED:
            self.predictor = TransformerDecoder(args["predictor"])
        
        if self.fusion_method=='fcooper':
            self.fusion_net = SpatialFusion()
        if self.fusion_method=='cobevt':
            self.fusion_net = SwapFusionEncoder(args['fusion_net'])
        if self.fusion_method=='where2comm':
            self.fusion_net = Where2comm(args['fusion_net'])
        if self.fusion_method=='v2xvit':
            self.fusion_net = V2XTransformer(args['fusion_net'])
            
        self.detect_head = detect_head(args=args['encoder_q']['args'])
        
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
            
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the query projection
        """
        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_q.data = param_q.data * self.momentum + param_k.data * (
                1.0 - self.momentum
            )

    def forward(self, data_dict):
        record_len = data_dict["record_len"]
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        spatial_correction_matrix = data_dict['spatial_correction_matrix'][:,:2,:,:]
        # B, max_cav, 3(dt dv infra), 1, 1
        prior_encoding = data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)[:,:2,:,:,:]

        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the query projection
            batch_dict_k = self.encoder_k(data_dict["processed_lidar_k"])
            batch_dict_q = self.encoder_q(data_dict["processed_lidar_q"])
            query=batch_dict_q["spatial_features_2d"]
            key=batch_dict_k["spatial_features_2d"]
            queryCopy = self.proj_q(query)
        

        key = self.compressor(key)
        key = self.proj_k(key)
        
        if self.USE_PRED:
            key = self.predictor(queryCopy, key)
        if self.PROJ_QUERY:
            query=queryCopy
            
        spatial_features_2d=[]
        for i in range(query.shape[0]):
            spatial_features_2d.append(query[i])
            spatial_features_2d.append(key[i])
        spatial_features_2d = torch.stack(spatial_features_2d,dim=0)
        assert len(record_len)*2==len(spatial_features_2d)
            
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
            prior_encoding = prior_encoding.repeat(1, 1, 1,
                                                regroup_feature.shape[3],
                                                regroup_feature.shape[4])
            regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)

            regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
            fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
            out = fused_feature.permute(0, 3, 1, 2)
            
        output_dict = self.detect_head(out)

        return output_dict

