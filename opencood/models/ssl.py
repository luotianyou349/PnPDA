import datetime
import torch
import torch.nn as nn
from opencood.models.sub_modules.downsample_conv import DownsampleConv

from opencood.tools import train_utils
from opencood.models.sub_modules.compressor import simple_align_v2,simple_align
from opencood.models.sub_modules.adapter import (
    TransformerDecoder,
    TransformerEncoder,
)


class ssl(nn.Module):
    def __init__(self, args) -> None:
        super(ssl, self).__init__()
        self.momentum = args["momentum"]

        self.encoder_q = train_utils.create_encoder(args["encoder_q"])
        self.encoder_k = train_utils.create_encoder(args["encoder_k"])
        self.use_shrink = False
        if 'shrink_header' in args:
            print('USE shrink_conv')
            self.use_shrink = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        self.compressor = simple_align_v2(args["compressor"])
        self.proj_q = TransformerEncoder(args["projector"])
        self.proj_k = TransformerEncoder(args["projector"])
        self.predictor = TransformerDecoder(args["predictor"])

        self.encoder_q = train_utils.load_pretrained_model(
            args["encoder_q"]["saved_pth"], self.encoder_q
        )
        self.encoder_k = train_utils.load_pretrained_model(
            args["encoder_k"]["saved_pth"], self.encoder_k
        )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_q.requires_grad = False
            param_k.requires_grad = False

        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_q.data.copy_(param_k.data)
            param_q.requires_grad = False

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
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the query projection
            batch_dict_k = self.encoder_k(data_dict["processed_lidar_k"])
            batch_dict_q = self.encoder_q(data_dict["processed_lidar_q"])
            query = self.proj_q(batch_dict_q["spatial_features_2d"])
            key = batch_dict_k['spatial_features_2d']

        if self.use_shrink:
            query = self.shrink_conv(query)
        key = self.compressor(key)
        key = self.proj_k(key)
        key = self.predictor(query, key)

        out_dict = {"features_q": query, "features_k": key}
        return out_dict
