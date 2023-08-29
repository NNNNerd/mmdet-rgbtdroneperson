# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector

import warnings

import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck

import torch.nn as nn


EPSILON = 1e-8

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
    def __init__(self, in_channels, conv1=True) -> None:
        super().__init__()
        if conv1:
            self.conv1x1 = nn.Conv2d(2*in_channels, in_channels, 1)
        else:
            self.conv1x1 = None
    
    def forward(self, en_ir, en_vi):
        temp = torch.cat((en_ir, en_vi), 1)
        if self.conv1x1 is not None:
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
    if spatial_type is 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type is 'sum':
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
        self.fusion_cat2 = Fusion_CAT(in_channels=in_channels, conv1=False)
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


@DETECTORS.register_module()
class FCOSHF(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(FCOSHF, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone_v = build_backbone(backbone)
        self.backbone_t = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        inchannels = neck['out_channels']
        self.fuse = Fusion_strategy(in_channels=inchannels)

        self.iter = 0

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        v_feat = self.backbone_v(img[0])
        t_feat = self.backbone_t(img[1])
        if self.with_neck:
            v_feat = self.neck(v_feat)
            t_feat = self.neck(t_feat)
        
        # import time
        # save_feature_to_img(v_feat, 'v_feat', int(time.time()))
        # save_feature_to_img(t_feat, 't_feat', int(time.time()))


        return self.fuse(v_feat, t_feat, 'cat')

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)

        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels

def save_feature_to_img(features, name, timestamp, channel=None, output_dir=None):
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import os

    if output_dir is None:
        output_dir = '/home/zy/mmdetection/work_dirs/fcosf_r50_caffe_fpn_gn-head_dp/vis'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not isinstance(timestamp, str):
        timestamp = str(timestamp)

    # for i in range(len(features)):
    if isinstance(features, list) or isinstance(features, tuple):
        for i in range(3):
            features_ = features[i]
            for j in range(features_.shape[0]):
                upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
                features_ = upsample(features_)

                feature = features_[j, :, :, :]
                if channel is None:
                    feature = torch.sum(feature, 0)
                else:
                    feature = feature[channel, :, :]
                feature = feature.detach().cpu().numpy() # 转为numpy

                # img = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) * 255 # 注意要防止分母为0！ 
                # img = img.astype(np.uint8)
                # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

                # plt.imshow(feature)
                # plt.axis('off')

                plt.matshow(feature, interpolation='nearest')
                plt.colorbar()
                plt.axis('off')

                dist_dir = os.path.join(output_dir, timestamp)
                if not os.path.exists(dist_dir):
                    os.mkdir(dist_dir)

                # cv2.imwrite(os.path.join(dist_dir, name + str(j) + '_' + timestamp + '.jpg'), img)

                plt.savefig(os.path.join(dist_dir, name + str(i) + '.png'))
                plt.close()
    else:
        for j in range(features.shape[0]):
            upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            features = upsample(features)

            feature = features[j, :, :, :]
            if channel is None:
                feature = torch.sum(feature, 0)
            else:
                feature = feature[channel, :, :]
            feature = feature.detach().cpu().numpy() # 转为numpy

            plt.matshow(feature, interpolation='nearest')
            plt.colorbar()
            plt.axis('off')

            dist_dir = os.path.join(output_dir, timestamp)
            if not os.path.exists(dist_dir):
                os.mkdir(dist_dir)

            plt.savefig(os.path.join(dist_dir, name + '.png'))
            plt.close()