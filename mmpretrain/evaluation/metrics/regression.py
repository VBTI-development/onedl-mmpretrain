# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from mmpretrain.evaluation.metrics.single_label import to_tensor
from mmpretrain.registry import METRICS


def calculate_distance(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    l_value: Union[int, str] = 1,
) -> torch.Tensor:
    """Calculate L-distance between pred and target.

    Args:
        pred (Union[torch.Tensor, np.ndarray]): prediction results
        target (Union[torch.Tensor, np.ndarray]): gt values
        l_value (int, optional): which L-value to calculate, see
            torch.linalg.norm. Defaults to 1.

    Returns:
        torch.Tensor: distance between pred and target.
    """
    assert isinstance(
        pred, (torch.Tensor,
               np.ndarray)), (f'The pred should be torch.Tensor or np.ndarray '
                              f'instead of {type(pred)}.')
    assert isinstance(
        target,
        (torch.Tensor,
         np.ndarray)), (f'The target should be torch.Tensor or np.ndarray '
                        f'instead of {type(target)}.')

    def to_tensor(array: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(array) if isinstance(array,
                                                     np.ndarray) else array

    # torch version is faster in most situations.
    pred = to_tensor(pred)
    target = to_tensor(target)
    return torch.linalg.norm(pred - target, ord=l_value,
                             dim=-1).mean()  # type: ignore[no-any-return]


@METRICS.register_module()
class RegressionMetric(BaseMetric):
    r"""A collection of l1 and l2 distance metrics for regression tasks.

    The collection of metrics is for regression.

    Args:
        items (Sequence[str]): The detailed metric items to evaluate, select
            from "l1_distance", "l2_distance"
            Defaults to ``('l1_distance', 'l2_distance')``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    Examples:
        >>> import torch
        >>> from onedl_mmclassification.regression.metrics import RegressionMetric
        >>> # -------------------- The Basic Usage --------------------
        >>> y_pred = [0., 1, 1, 3]
        >>> y_true = [0., 4, 2, 3]
        >>> # Output l1 and l2 distance metrics
        >>> RegressionMetric.calculate(y_pred, y_true)
        (tensor(4.), tensor(3.1623))
        >>> # ------------------- Use with Evaluator -------------------
        >>> from mmpretrain.structures import DataSample
        >>> from mmengine.evaluator import Evaluator
        >>> data_samples = [
        ...     DataSample().set_gt_label([0, i%5]).set_pred_score(torch.rand(2))
        ...     for i in range(1000)
        ... ]
        >>> evaluator = Evaluator(metrics=RegressionMetric())
        >>> evaluator.process(data_samples)
        >>> evaluator.evaluate(1000)
        {'regression/l1_distance': 2.1867,
         'regression/l2_distance': 1.8436}
    """  # noqa: E501

    default_prefix: Optional[str] = 'regression'

    def __init__(
        self,
        items: Sequence[str] = ('l1_distance', 'l2_distance'),
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        for item in items:
            assert item in [
                'l1_distance', 'l2_distance'
            ], (f'The metric {item} is not supported by `RegressionMetric`,'
                ' please specify from "l1_distance", "l2_distance".')
        self.items = tuple(items)

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()
            result['pred_score'] = data_sample['pred_score'].cpu()

            result['gt_score'] = data_sample['gt_score'].cpu()
            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        # NOTICE: don't access `self.results` from the method. `self.results`
        # are a list of results from multiple batch, while the input `results`
        # are the collected results.

        def pack_results(l1_distance: Any, l2_distance: Any) -> Dict[str, Any]:
            single_metrics = {}
            if 'l1_distance' in self.items:
                single_metrics['l1_distance'] = l1_distance
            if 'l2_distance' in self.items:
                single_metrics['l2_distance'] = l2_distance
            return single_metrics

        # concat
        target = torch.stack([res['gt_score'] for res in results])
        pred = torch.stack([res['pred_score'] for res in results])
        metrics = self.calculate(pred, target)
        return pack_results(*metrics)

    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray,
                    Sequence], target: Union[torch.Tensor, np.ndarray,
                                             Sequence]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Calculate the l1 and l2 distance.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (torch.Tensor | np.ndarray | Sequence): The target of
                each prediction with shape (N, ).

        Returns:
            Tuple: The tuple contains  l1 and l2 distance.
            And the type of each item is:

            - torch.Tensor: The shape is (C, ) where C is the number of values.
        """
        pred_tensor = to_tensor(pred)
        target_tensor = to_tensor(target).to(torch.int64)
        assert pred_tensor.size(0) == target_tensor.size(0), (
            f"The size of pred ({pred_tensor.size(0)}) doesn't match "
            f'the target ({target_tensor.size(0)}).')

        results = [
            calculate_distance(pred_tensor, target_tensor, 1),
            calculate_distance(pred_tensor, target_tensor, 2),
        ]
        return results
