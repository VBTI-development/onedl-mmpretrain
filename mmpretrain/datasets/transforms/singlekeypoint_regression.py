# Copyright (c) VBTI. All rights reserved.
# mypy: ignore-errors
import numpy as np
from mmcv.transforms import BaseTransform

from mmpretrain.datasets.transforms.processing import Albumentations
from mmpretrain.registry import TRANSFORMS

try:
    import albumentations
except ImportError:
    albumentations = None


@TRANSFORMS.register_module()
class DenormalizeKeypointLocation(BaseTransform):
    """Denormalize keypoint location (xy) to image size (reverse effect of
    normalization)"""
    def __init__(self) -> None:
        pass

    def transform(self, results):
        h, w = results['img'].shape[:2]
        if 'gt_score' in results:
            results['gt_score'] = [
                x * y for x, y in zip(results['gt_score'], (w, h))
            ]
        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        return self.__class__.__name__ + '()'


@TRANSFORMS.register_module()
class NormalizeKeypointLocation(BaseTransform):
    """Normalize keypoint location (xy) to image size.

    WARNING: This will mess up the evaluation, since this is not done
    for eval results.
    """
    def __init__(self) -> None:
        pass

    def transform(self, results):
        h, w = results['img'].shape[:2]
        if 'gt_score' in results:
            results['ori_kp'] = results['gt_score']
            results['gt_score'] = [
                x / y for x, y in zip(results['gt_score'], (w, h))
            ]
        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        return self.__class__.__name__ + '()'


@TRANSFORMS.register_module()
class AlbuKeypoint(Albumentations):
    """Albumentation augmentation.

    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.
    An example of ``transforms`` is as followed:
    .. code-block::
        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]
    Args:
        transforms (list[dict]): A list of albu transformations
        keymap (dict): Contains {'input key':'albumentation-style key'}
    """
    def __init__(self,
                 transforms,
                 keymap=None,
                 update_pad_shape=False,
                 keypoint_params=None):
        if albumentations is None:
            raise RuntimeError('albumentations is not installed')
        else:
            from albumentations import Compose

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape

        if keypoint_params is not None:
            keypoint_params = albumentations.KeypointParams(**keypoint_params)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           keypoint_params=keypoint_params,
                           strict=False)

        if not keymap:
            self.keymap_to_albu = {'img': 'image'}
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def transform(self, results: dict) -> dict:
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        has_keypoints = 'keypoints' in results

        if has_keypoints:
            results['keypoints'] = [
                results['keypoints']
            ]  # albu expects a list of keypoints, not a single keypoint
        else:
            results['keypoints'] = []

        results = self.aug(**results)

        if has_keypoints:
            if isinstance(results['keypoints'], list):
                results['keypoints'] = np.array(
                    results['keypoints']).squeeze()  # unpack
        else:
            results.pop('keypoints')

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results
