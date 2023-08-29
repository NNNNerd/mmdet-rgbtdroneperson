import copy
import inspect

import mmcv
import numpy as np
from numpy import random

from mmdet.core import PolygonMasks
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..builder import PIPELINES

@PIPELINES.register_module()
class MultiNormalize:
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean_list, std_list, to_rgb=True):
        self.mean = [np.array(mean, dtype=np.float32) for mean in mean_list]
        self.std = [np.array(std, dtype=np.float32) for std in std_list]
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for i, key in enumerate(results.get('img_fields', ['img1', 'img2'])):
            results[key] = mmcv.imnormalize(results[key], self.mean[i], self.std[i], 
                                            self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class RandomMasking:
    def __init__(self, p=(0.25, 0.25, 0.5)):
        # probabilities of masking RGB, T and nothing
        self.p = p

    def __call__(self, results):
        sample_mode = (0, 1, 2)
        mode = random.choice(sample_mode, p=self.p)
        if mode == 0 or mode == 1:
            key = results.get('img_fields', ['img1', 'img2'])[mode]
            results[key] = np.zeros(results[key].shape)
        return results


@PIPELINES.register_module()
class SpectralShift:
    # shift visible images
    def __init__(self, shift_ratio_range=(0, 0.05)):
        self.min_ratio, self.max_ratio = shift_ratio_range

    def __call__(self, results):
        img = results['img1']
        h, w, c = img.shape
        y_ratio = random.uniform(self.min_ratio, self.max_ratio)
        x_ratio = random.uniform(self.min_ratio, self.max_ratio)
        delta_y = int(h * y_ratio)
        delta_x = int(w * x_ratio)
        # expand and crop
        expand_img = np.full((h+2*delta_y, w+2*delta_x, c), 255, dtype=img.dtype)        
        expand_img[delta_y:delta_y+h, delta_x:delta_x+w] = img
        direction = random.choice((0, 1, 2, 3))
        if direction == 0:
            img = expand_img[:h, :w]
        elif direction == 1:
            img = expand_img[:h, 2*delta_x:]
        elif direction == 2:
            img = expand_img[2*delta_y:, 2*delta_x:]
        else:
            img = expand_img[2*delta_y:, :w]
        results['img1'] = img
        return results