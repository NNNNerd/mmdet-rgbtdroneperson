# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS, build_backbone, build_head
from .single_stage import SingleStageDetector
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import bbox2result, multi_apply


@DETECTORS.register_module()
class QFDet(SingleStageDetector):
    """Implementation of `QFDet`."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_prehead,
                 base_fusion='cat',
                 quality_attention=True,
                 poolupsample=None,
                 reweight=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(QFDet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
        self.backbone_t = build_backbone(backbone)
        self.base_fusion = base_fusion
        self.fuse = Fusion_strategy(neck['out_channels'])
        bbox_prehead.update(train_cfg=train_cfg)
        bbox_prehead.update(test_cfg=test_cfg)
        self.bbox_prehead = build_head(bbox_prehead)
        self.iter = 0
        if poolupsample is not None:
            self.poolupsample = PoolingUpsample(neck['out_channels'])
        else:
            self.poolupsample = None
        self.quality_attention = quality_attention

        self.reweight = reweight

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        # self.iter = self.iter + 1
        v_img, t_img = img
        v_feats = self.backbone(v_img)
        t_feats = self.backbone_t(t_img)
        if self.with_neck:
            v_feats = self.neck(v_feats)
            t_feats = self.neck(t_feats)

        return (v_feats, t_feats)

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        x_vs, x_ts = x
        num_level = len(x_vs)
        quality_preds_t, quality_preds_v = self.bbox_prehead.forward_test(x)
        fused_x = []
        for i in range(num_level):
            quality_pred_t, quality_pred_v = my_norm(quality_preds_t[i], quality_preds_v[i], type='minmax')
            
            x_t = (1 + quality_pred_t) * x_ts[i]
            x_v = (1 + quality_pred_v) * x_vs[i]

            if self.poolupsample is not None and i < num_level-1:
                x_v = self.poolupsample(x_v)
            
            fused_x_ = self.fuse(x_t, x_v, 'cat')
            fused_x.append(fused_x_)
        outs = self.bbox_head(fused_x)
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
        # super(SingleStageDetector, self).forward_train(img, img_metas)
        losses = dict()
        x = self.extract_feat(img)
        x_vs, x_ts = x

        cls_scores_t, bbox_preds_t, centernesses_t, quality_preds_t, \
        cls_scores_v, bbox_preds_v, centernesses_v, quality_preds_v = self.bbox_prehead(x)
        
        pre_loss = self.bbox_prehead.loss(
             cls_scores_t, 
             bbox_preds_t, 
             centernesses_t, 
             quality_preds_t,
             cls_scores_v, 
             bbox_preds_v, 
             centernesses_v, 
             quality_preds_v,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None)
        losses.update(pre_loss)
        
        fused_x = self.qce_fusion(x, quality_preds_t, quality_preds_t)

        loss = self.bbox_head.forward_train(fused_x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        if self.reweight:
            weight = torch.linspace(1, 5, steps=len(x_ts), device=loss['loss_cls'][0].device)
            for k in loss.keys():
                for i in range(weight.size()[0]):
                    loss[k][i] = weight[i]*loss[k][i]
        losses.update(loss)
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
        self.iter += 1
        x = self.extract_feat(img)

        quality_preds_t, quality_preds_v = self.bbox_prehead.forward_test(x)
        
        fused_x = self.qce_fusion(x, quality_preds_t, quality_preds_v)

        results_list = self.bbox_head.simple_test(
            fused_x, img_metas, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results
    
    def qce_fusion(self, x, quality_t, quality_v):
        x_vs, x_ts = x

        num_level = len(x_vs)

        fused_x = []
        for i in range(num_level):
            x_t = x_ts[i]
            x_v = x_vs[i]
            if self.quality_attention:
                quality_pred_t = torch.max(quality_t[i], dim=1, keepdim=True)[0]
                quality_pred_v = torch.max(quality_v[i], dim=1, keepdim=True)[0]

                quality_pred_t, quality_pred_v = my_norm(quality_pred_t, quality_pred_v, type='minmax')
                
                x_t = (1 + quality_pred_t) * x_t
                x_v = (1 + quality_pred_v) * x_v            
            
            if self.poolupsample is not None and i < num_level-1:
                x_v = self.poolupsample(x_v)
                # x_t = self.poolupsample(x_t)

            fused_x_ = self.fuse(x_t, x_v, self.base_fusion)

            fused_x.append(fused_x_)

        return fused_x


def save_feature_to_img(features, name, timestamp, method='cv2', channel=None, output_dir=None, maxmin=None):
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import os

    if output_dir is None:
        output_dir = '/home/zhangyan22/mmdetection/work_dirs/atss_r50_fpn_hf_qprelchead_1x_dp/vis_wo_optimize_v'
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

                dist_dir = os.path.join(output_dir, timestamp)
                if not os.path.exists(dist_dir):
                    os.mkdir(dist_dir)

                if method == 'cv2':
                    if maxmin is not None:
                        img = (feature - maxmin[1])/(maxmin[0] - maxmin[1] + 1e-5) * 255
                    else:
                        img = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) * 255 # 注意要防止分母为0！ 
                    img = img.astype(np.uint8)
                    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                    plt.imshow(feature)
                    plt.axis('off')
                    cv2.imwrite(os.path.join(dist_dir, name + str(i) + '.jpg'), img)

                elif method == 'matshow':
                    plt.matshow(feature, interpolation='nearest')
                    plt.colorbar()
                    plt.axis('off')

                    plt.savefig(os.path.join(dist_dir, name + str(i) + '.png'))
                    plt.close()
                else:
                    NotImplementedError()
    
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

            dist_dir = os.path.join(output_dir, timestamp)
            if not os.path.exists(dist_dir):
                os.mkdir(dist_dir)

            if method == 'cv2':
                if maxmin is not None:
                    img = (feature - maxmin[1])/(maxmin[0] - maxmin[1] + 1e-5) * 255
                else:
                    img = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) * 255 # 注意要防止分母为0！ 
                img = img.astype(np.uint8)
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                plt.imshow(feature)
                plt.axis('off')
                cv2.imwrite(os.path.join(dist_dir, name + '.jpg'), img)

            elif method == 'matshow':
                plt.matshow(feature, interpolation='nearest')
                plt.colorbar()
                plt.axis('off')

                plt.savefig(os.path.join(dist_dir, name + '.png'))
                plt.close()
            else:
                NotImplementedError()


def feature_map_norm(x):
    # input (B, C, H, W)
    bs, c, h, w = x.shape
    x = x.view(bs, -1, h*w)
    x_mean = torch.mean(x, dim=2, keepdim=True)
    x_std = torch.std(x, dim=2, keepdim=True)
    x = (x - x_mean) / x_std
    x = x.view(bs, c, h, w)
    return x

def my_norm(x1, x2, type='standard'):
    assert type in ['standard', 'minmax']
    bs, _ , H, W = x1.size()
    _, _, h, w = x2.size()
    x1 = x1.view(bs, -1, H*W)
    x2 = x2.view(bs, -1, h*w)
    concat = torch.cat((x1, x2), dim=2)
    if type == 'standard':
        x_mean = torch.mean(concat, dim=2, keepdim=True)
        x_std = torch.std(concat, dim=2, keepdim=True)
        x1 = (x1 - x_mean) / x_std
        x2 = (x2 - x_mean) / x_std
    elif type == 'minmax':
        x_min = torch.min(concat, dim=2, keepdim=True)[0]
        x_max = torch.max(concat, dim=2, keepdim=True)[0]
        x1 = (x1 - x_min) / x_max
        x2 = (x2 - x_min) / x_max
    x1 = x1.view(bs, -1, H, W)
    x2 = x2.view(bs, -1, h, w)
    return [x1, x2]


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

# Fusion strategy, two type
class Fusion_strategy(nn.Module):
    def __init__(self, in_channels):
        super(Fusion_strategy, self).__init__()
        self.fusion_add = Fusion_ADD()
        self.fusion_avg = Fusion_AVG()
        self.fusion_max = Fusion_MAX()
        self.fusion_cat = Fusion_CAT(in_channels=in_channels)

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
        if isinstance(v_feat, tuple) or isinstance(v_feat, list):
            fused_feat = []
            for i in range(len(v_feat)):
                fused_feat.append(fusion_operation(v_feat[i], t_feat[i]))
        else:
            fused_feat = fusion_operation(v_feat, t_feat)
        
        return fused_feat



class PoolingUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.maxpooling = nn.MaxPool2d(2, 2, dilation=1)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1)
    
    def forward(self, x):
        x_ = self.maxpooling(x)
        # x_ = self.upsample(x_)
        x_ = F.interpolate(x_, mode='bilinear', size=x.shape[-2:], align_corners=True)
        # import pdb; pdb.set_trace()
        x = self.conv1x1(torch.cat((x, x_), 1))
        return x