import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.tools import train_utils
from opencood.models.sub_modules.adapter import projector
from opencood.models.sub_modules.point_pillar_predictor import predictor
from opencood.models.sub_modules.point_pillar_encoder import PointPillarEncoder
# from opencood.models.fuse_modules.maxout import Maxout
from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion

    
class cls_head(nn.Module):
    def __init__(self,args) -> None:
        super(cls_head,self).__init__()
        n_layer = len(args["base_bev_backbone"]["layer_nums"])
        self.cls_head=nn.Conv2d(
            128 * n_layer, args["anchor_number"], kernel_size=1
        )
    
    def forward(self,x):
        return self.cls_head(x)
    
class reg_head(nn.Module):
    def __init__(self,args) -> None:
        super(reg_head,self).__init__()
        n_layer = len(args["base_bev_backbone"]["layer_nums"])
        self.reg_head=nn.Conv2d(128 * n_layer, 7 * args["anchor_number"], kernel_size=1)
    
    def forward(self,x):
        return self.reg_head(x)

class PointPillarCPM(nn.Module):
    def __init__(self, args):
        super(PointPillarCPM, self).__init__()

        self.encoder_q = PointPillarEncoder(args["encoder_q"])
        self.encoder_k = PointPillarEncoder(args["encoder_k"])

        self.projector = projector(args["projector"])
        self.predictor = predictor(args["predictor"])
        # self.fusion_net = Maxout(args["fuse_net"])
        self.fusion_net = SpatialFusion()

        self.cls_head = cls_head(args["encoder_q"])
        self.reg_head = reg_head(args["encoder_q"])

        self.encoder_q = train_utils.load_pretrained_model(
            args["encoder_q"]["saved_pth"], self.encoder_q
        )
        self.encoder_k = train_utils.load_pretrained_model(
            args["encoder_k"]["saved_pth"], self.encoder_k
        )
        self.projector = train_utils.load_pretrained_model(
            args["projector"]["saved_pth"], self.projector
        )
        self.predictor = train_utils.load_pretrained_model(
            args["predictor"]["saved_pth"], self.predictor
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
            # param_cls.requires_grad = False
            # param_reg.requires_grad = False

    def forward(self, data_dict):
        record_len = data_dict["record_len"]
        with torch.no_grad():
            batch_dict_q = self.encoder_q(data_dict["processed_lidar_q"])
            batch_dict_k = self.encoder_k(data_dict["processed_lidar_k"])

        query = batch_dict_q["spatial_features_2d"]
        key = batch_dict_k["spatial_features_2d"]

        key = self.projector(query, key)
        key = self.predictor(key)
        # b,c,h,w
        key = key.permute(0, 3, 1, 2).contiguous()
        
        # n,c,h,w
        spatial_features_2d = torch.cat((query,key),dim=0)
        out = self.fusion_net(spatial_features_2d, record_len)
        # out = query

        psm = self.cls_head(out)
        rm = self.reg_head(out)

        output_dict = {"psm": psm, "rm": rm}

        return output_dict
