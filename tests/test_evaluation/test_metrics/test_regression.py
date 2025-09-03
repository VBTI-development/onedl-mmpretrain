# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch

from mmpretrain.evaluation.metrics.regression import (RegressionMetric,
                                                      calculate_distance)


class TestRegressionMetric(TestCase):
    def test_calculate_distance(self):
        # Test L1 and L2 distance with numpy arrays
        pred = np.array([0., 1, 1, 3])
        target = np.array([0., 4, 2, 3])
        l1 = calculate_distance(pred, target, 1)
        l2 = calculate_distance(pred, target, 2)
        self.assertIsInstance(l1, torch.Tensor)
        self.assertIsInstance(l2, torch.Tensor)
        self.assertAlmostEqual(l1.item(), 4.0, places=4)
        self.assertAlmostEqual(l2.item(), 3.1623, places=4)

        # Test with torch tensors
        pred = torch.tensor([0., 1, 1, 3])
        target = torch.tensor([0., 4, 2, 3])
        l1 = calculate_distance(pred, target, 1)
        l2 = calculate_distance(pred, target, 2)
        self.assertAlmostEqual(l1.item(), 4.0, places=4)
        self.assertAlmostEqual(l2.item(), 3.1623, places=4)

    def test_metric_calculate(self):
        # Test RegressionMetric.calculate static method
        pred = [0., 1, 1, 3]
        target = [0., 4, 2, 3]
        l1, l2 = RegressionMetric.calculate(pred, target)
        self.assertIsInstance(l1, torch.Tensor)
        self.assertIsInstance(l2, torch.Tensor)
        self.assertAlmostEqual(l1.item(), 4.0, places=4)
        self.assertAlmostEqual(l2.item(), 3.1623, places=4)

        # Test with torch tensors
        pred = torch.tensor([0., 1, 1, 3])
        target = torch.tensor([0., 4, 2, 3])
        l1, l2 = RegressionMetric.calculate(pred, target)
        self.assertAlmostEqual(l1.item(), 4.0, places=4)
        self.assertAlmostEqual(l2.item(), 3.1623, places=4)

    def test_process_and_compute_metrics(self):
        # Simulate a batch of data samples
        data_samples = [
            {
                'pred_score': torch.tensor([1.0, 2.0]),
                'gt_score': torch.tensor([1.5, 2.5])
            },
            {
                'pred_score': torch.tensor([3.0, 4.0]),
                'gt_score': torch.tensor([2.0, 5.0])
            },
        ]
        metric = RegressionMetric()
        metric.process(None, data_samples)
        results = metric.results
        self.assertEqual(len(results), 2)
        computed = metric.compute_metrics(results)
        self.assertIn('l1_distance', computed)
        self.assertIn('l2_distance', computed)
        self.assertIsInstance(computed['l1_distance'], torch.Tensor)
        self.assertIsInstance(computed['l2_distance'], torch.Tensor)

    def test_items_argument(self):
        # Test selecting only l1_distance
        pred = [0., 1, 1, 3]
        target = [0., 4, 2, 3]
        metric = RegressionMetric(items=['l1_distance'])
        metric.process(None, [{
            'pred_score': torch.tensor(pred),
            'gt_score': torch.tensor(target)
        }])
        computed = metric.compute_metrics(metric.results)
        self.assertIn('l1_distance', computed)
        self.assertNotIn('l2_distance', computed)

        # Test selecting only l2_distance
        metric = RegressionMetric(items=['l2_distance'])
        metric.process(None, [{
            'pred_score': torch.tensor(pred),
            'gt_score': torch.tensor(target)
        }])
        computed = metric.compute_metrics(metric.results)
        self.assertNotIn('l1_distance', computed)
        self.assertIn('l2_distance', computed)

    def test_invalid_items(self):
        # Test invalid metric item
        with self.assertRaises(AssertionError):
            RegressionMetric(items=['invalid_metric'])
