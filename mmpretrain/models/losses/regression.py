# Copyright (c) VBTI. All rights reserved.
from typing import Literal, Optional

from torch import Tensor, nn

from mmpretrain.models import LOSSES
from mmpretrain.models.losses.utils import weight_reduce_loss


@LOSSES.register_module()
class RegressionLoss(nn.Module):
    """Module to calculate loss for regression values."""
    def __init__(self,
                 loss_type: Literal['L1Loss', 'L2Loss', 'MSELoss'],
                 reduction: str = 'mean',
                 loss_weight: float = 1.0):
        """Module to calculate loss for regression values.

        Args:
            loss_type: which loss is calculated, one of 'L1Loss',
                'L2Loss'. 'MSELoss'
            reduction: how to reduce the loss, same as pytorch
            loss_weight: multiplication factor
        """
        super().__init__()
        self.loss_type = loss_type

        self.reduction = reduction
        self.loss_weight = loss_weight

        if loss_type == 'L1Loss':
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type in ('L2Loss', 'MSELoss'):
            self.criterion = nn.MSELoss(reduction='none')
        else:
            raise ValueError(
                "Loss type should be one of 'L1Loss', 'L2Loss'. 'MSELoss'")

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[str] = None):  # noqa: PLR0913
        """Forward function to calculate accuracy.

        Args:
            pred: Prediction of models.
            target: Target for each prediction.
            weight: element-wise weights
            avg_factor: average factor when computing the mean of losses
            reduction_override: override reduction method.
        Returns:
            list[torch.Tensor]: The loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)

        loss = self.loss_weight * self.criterion(pred, target)
        loss = weight_reduce_loss(loss,
                                  weight=weight,
                                  reduction=reduction,
                                  avg_factor=avg_factor)
        return loss
