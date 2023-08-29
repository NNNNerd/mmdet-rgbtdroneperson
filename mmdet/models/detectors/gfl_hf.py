# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS, build_backbone
from .single_stage import SingleStageDetector
import torch
import torch.nn as nn


@DETECTORS.register_module()
class GFLHF(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(GFLHF, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained, init_cfg)
        self.backbone_t = build_backbone(backbone)
        self.fuse = Fusion_strategy(neck['out_channels'])

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        # self.iter = self.iter + 1
        v_img, t_img = img
        v_feat = self.backbone(v_img)
        t_feat = self.backbone_t(t_img)
        if self.with_neck:
            v_feat = self.neck(v_feat)
            t_feat = self.neck(t_feat)
        features = []
        for i in range(len(v_feat)):
            feat = self.fuse(v_feat[i], t_feat[i], 'cat')
            features.append(feat)
        return features

class Fusion_ADD(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = en_ir + en_vi
        return temp

class Fusion_AVG(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = (en_ir + en_vi) / 2
        return temp

class Fusion_MAX(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = torch.max(en_ir, en_vi)
        return temp

class Fusion_CAT(torch.nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(2*in_channels, in_channels, 1)
    
    def forward(self, en_ir, en_vi):
        temp = torch.cat((en_ir, en_vi), 1)
        temp = self.conv1x1(temp)
        return temp

class Fusion_GATED(torch.nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.t_conv1x1 = nn.Conv2d(in_channels, in_channels, 1)
        self.v_conv1x1 = nn.Conv2d(in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, t_feat, v_feat):
        t_gate = self.sigmoid(self.t_conv1x1(t_feat))
        v_gate = self.sigmoid(self.v_conv1x1(v_feat))
        return t_gate*t_feat + v_gate*v_feat

EPSILON = 1e-10

class Fusion_SPA(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        shape = en_ir.size()
        spatial_type = 'mean'
        # calculate spatial attention
        spatial1 = spatial_attention(en_ir, spatial_type)
        spatial2 = spatial_attention(en_vi, spatial_type)
        # get weight map, soft-max
        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
        spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

        spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
        spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
        tensor_f = spatial_w1 * en_ir + spatial_w2 * en_vi
        return tensor_f

# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    spatial = []
    if spatial_type == 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type == 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial

# Fusion strategy, two type
class Fusion_strategy(nn.Module):
    def __init__(self, in_channels):
        super(Fusion_strategy, self).__init__()
        self.fusion_add = Fusion_ADD()
        self.fusion_avg = Fusion_AVG()
        self.fusion_max = Fusion_MAX()
        self.fusion_cat = Fusion_CAT(in_channels=in_channels)
        self.fusion_cat2 = Fusion_CAT(in_channels=in_channels)
        self.fusion_spa = Fusion_SPA()
        self.fusion_gated = Fusion_GATED(in_channels=in_channels)

    def forward(self, v_feat, t_feat, fs_type):
        self.fs_type = fs_type
        if self.fs_type == 'add':
            fusion_operation = self.fusion_add
        elif self.fs_type == 'avg':
            fusion_operation = self.fusion_avg
        elif self.fs_type == 'max':
            fusion_operation = self.fusion_max
        elif self.fs_type == 'cat':
            fusion_operation = self.fusion_cat
        elif self.fs_type == 'cat2':
            fusion_operation = self.fusion_cat2
        elif self.fs_type == 'spa':
            fusion_operation = self.fusion_spa
        elif self.fs_type == 'gated':
            fusion_operation = self.fusion_gated
        if isinstance(v_feat, tuple) or isinstance(v_feat, list):
            fused_feat = []
            for i in range(len(v_feat)):
                fused_feat.append(fusion_operation(v_feat[i], t_feat[i]))
        else:
            fused_feat = fusion_operation(v_feat, t_feat)
        
        return fused_feat
