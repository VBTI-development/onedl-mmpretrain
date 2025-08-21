# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
from torch import nn

from mmpretrain.models.builder import HEADS
from mmpretrain.models.heads import ClsHead
from mmpretrain.structures import DataSample


@HEADS.register_module()
class RegressionHead(ClsHead):
    """RegressionHead head.

    Args:
        loss (dict): Config of regression loss.
    """
    def __init__(self,
                 loss: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if loss is None:
            loss = dict(type='RegressionLoss',
                        loss_type='L1Loss',
                        loss_weight=1.0)
        super(RegressionHead, self).__init__(init_cfg=init_cfg,
                                             loss=loss,
                                             cal_acc=False)

    def _get_predictions(self, cls_score: torch.Tensor,
                         data_samples: list[DataSample]) -> list[DataSample]:
        """Post-process the output of head.

        Sets the score and prediction label for each prediction in the
        data sample.

        Args:
            cls_score: The output score of the head.
            data_samples: The data samples to be processed.
        Returns:
            list[DataSample]: The post-processed data samples.
        """
        pred_scores = cls_score.detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score in zip(data_samples, pred_scores):
            if data_sample is None:
                data_sample = DataSample()  # noqa: PLW2901

            data_sample.set_pred_score(score).set_pred_label(score)
            out_data_samples.append(data_sample)
        return out_data_samples


@HEADS.register_module()
class LinearRegressionHead(RegressionHead):
    """Linear regression head.

    Args:
        num_classes: Number of outputs.
        in_channels: Number of channels in the input feature map.
        init_cfg: The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 init_cfg: Optional[dict] = None,
                 *args,
                 **kwargs):
        if init_cfg is None:
            init_cfg = dict(type='Normal', layer='Linear', std=0.01)
        super(LinearRegressionHead, self).__init__(*args,
                                                   **kwargs,
                                                   init_cfg=init_cfg)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        Args:
            feats: The input ``feats`` is a tuple of tensor, and each tensor is
                the feature of a backbone stage. In ``LinearClsHead``, we just
                obtain the feature of the last stage.
        """
        # The LinearClsHead doesn't have other module, just return after
        # unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process.

        Args:
            feats: The input ``feats`` is a tuple of tensor, and each tensor is
                the feature of a backbone stage. In ``LinearClsHead``, we just
                obtain the feature of the last stage.
        """

        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.fc(pre_logits)
        return cls_score
