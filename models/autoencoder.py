#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Point cloud auto-encoder backbone
'''

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import sys
import os

# neural atlas
import pytorch3d.ops

USE_CUDA = True

from . import get_model_class
from .tearingnet import TearingNetBasicModel
from .tearingnet_graph import TearingNetGraphModel


class PointCloudAutoencoder(nn.Module):

    def __init__(self, opt):
        super(PointCloudAutoencoder, self).__init__()
        
        encoder_class = get_model_class(opt.encoder)
        self.encoder = encoder_class(opt)
        decoder_class = get_model_class(opt.decoder)
        self.decoder = decoder_class(opt)
        self.is_train = (opt.phase.lower() == 'train')
        if self.is_train:
            self.xyz_loss_type = opt.xyz_loss_type
        self.xyz_chamfer_weight = opt.xyz_chamfer_weight

    def forward(self, data):

        cw = self.encoder(data)
        if isinstance(self.decoder, TearingNetBasicModel):
            rec0, rec1, grid = self.decoder(cw)
            return {"rec": rec1, "rec_pre": rec0, "grid": grid, "cw": cw}
        elif isinstance(self.decoder, TearingNetGraphModel):
            rec0, rec1, rec2, grid, graph_wght = self.decoder(cw)
            return {"rec": rec2, "rec_pre": rec1, "rec_pre2": rec0, "grid": grid, "graph_wght": graph_wght, "cw": cw}
        else:
            rec = self.decoder(cw)
            return {"rec": rec, "cw": cw}

    def xyz_loss(self, data, rec, xyz_loss_type=-1):

        if xyz_loss_type == -1:
            xyz_loss_type = self.xyz_loss_type
        dist1, _, _ = pytorch3d.ops.knn_points(data, rec, K=1)                  # (N, P1, 1)
        dist1 = dist1.squeeze(dim=2)                                            # (N, P1)
        dist2, _, _ = pytorch3d.ops.knn_points(rec, data, K=1)                  # (N, P2, 1)
        dist2 = dist2.squeeze(dim=2)                                            # (N, P2)

        dist2 = dist2 * self.xyz_chamfer_weight

        # Different variants of the Chamfer distance
        if xyz_loss_type == 0: # augmented Chamfer distance
            loss = torch.max(torch.mean(torch.sqrt(dist1), 1), torch.mean(torch.sqrt(dist2), 1))
            loss = torch.mean(loss)
        elif xyz_loss_type == 1:
            loss = torch.mean(torch.sqrt(dist1), 1) + torch.mean(torch.sqrt(dist2), 1)
            loss = torch.mean(loss)
        elif xyz_loss_type == 2: # used in other papers
            loss = torch.mean(dist1) + torch.mean(dist2)
        return loss


if __name__ == '__main__':
    USE_CUDA = True