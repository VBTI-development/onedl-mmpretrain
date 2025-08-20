# Copyright (c) VBTI. All rights reserved.
import copy
from unittest import TestCase

import numpy as np

from mmpretrain.registry import TRANSFORMS


def construct_toy_data():
    img = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                   dtype=np.uint8)
    img = np.stack([img, img, img], axis=-1)
    results = dict()
    # image
    results['ori_img'] = img
    results['img'] = copy.deepcopy(img)
    results['ori_shape'] = img.shape
    results['img_shape'] = img.shape
    return results


class TestDenormalizeKeypointLocation(TestCase):
    def test_transform(self):
        results = dict(img=np.random.randint(0, 256, (256, 384, 3), np.uint8),
                       gt_score=[0.5, 0.5])

        # test random crop by default.
        cfg = dict(type='DenormalizeKeypointLocation')
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertListEqual(results['gt_score'], [192, 128])

    def test_repr(self):
        cfg = dict(type='DenormalizeKeypointLocation')
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(repr(transform), 'DenormalizeKeypointLocation()')

    def test_transform_round_robin(self):
        results = dict(img=np.random.randint(0, 256, (256, 384, 3), np.uint8),
                       gt_score=[0.5, 0.5])

        # test random crop by default.
        cfg = dict(type='DenormalizeKeypointLocation')
        transform = TRANSFORMS.build(cfg)
        results = transform(results)

        # test random crop by default.
        cfg = dict(type='NormalizeKeypointLocation')
        transform2 = TRANSFORMS.build(cfg)
        results = transform2(results)

        self.assertListEqual(results['gt_score'], [0.5, 0.5])


class TestNormalizeKeypointLocation(TestCase):
    def test_transform(self):
        results = dict(img=np.random.randint(0, 256, (256, 384, 3), np.uint8),
                       gt_score=[192, 128])

        # test random crop by default.
        cfg = dict(type='NormalizeKeypointLocation')
        transform = TRANSFORMS.build(cfg)
        results = transform(results)
        self.assertListEqual(results['gt_score'], [0.5, 0.5])

    def test_repr(self):
        cfg = dict(type='NormalizeKeypointLocation')
        transform = TRANSFORMS.build(cfg)
        self.assertEqual(repr(transform), 'NormalizeKeypointLocation()')

    def test_transform_round_robin(self):
        results = dict(img=np.random.randint(0, 256, (256, 384, 3), np.uint8),
                       gt_score=[192, 128])

        # test random crop by default.
        cfg = dict(type='NormalizeKeypointLocation')
        transform = TRANSFORMS.build(cfg)
        results = transform(results)

        # test random crop by default.
        cfg = dict(type='DenormalizeKeypointLocation')
        transform2 = TRANSFORMS.build(cfg)
        results = transform2(results)

        self.assertListEqual(results['gt_score'], [192, 128])
