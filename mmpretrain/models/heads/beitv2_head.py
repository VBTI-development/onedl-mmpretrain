# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class BEiTV2Head(BaseModule):
    """Head for BEiT v2 Pre-training.

    Compute the logits and the cross entropy loss.

    Args:
        embed_dims (int): The dimension of embedding.
        num_embed (int): The number of classification types.
        loss (dict): The config of loss.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Defaults to None.
    """
    def __init__(
        self,
        embed_dims: int,
        num_embed: int,
        loss: dict,
        init_cfg: Optional[Union[dict, List[dict]]] = dict(type='TruncNormal',
                                                           layer='Linear',
                                                           std=0.02,
                                                           bias=0)
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.cls_head = nn.Linear(embed_dims, num_embed)
        self.loss_module = MODELS.build(loss)

    def loss(self, feats: torch.Tensor, feats_cls_pt: torch.Tensor,
             target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Generate loss.

        Args:
            feats (torch.Tensor): Features from backbone.
            feats_cls_pt (torch.Tensor) : Features from class late layers for
                pretraining.
            target (torch.Tensor): Target generated by target_generator.
            mask (torch.Tensor): Generated mask for pretraing.
        """
        mask = mask.flatten(1).to(torch.bool)
        target = target[mask]

        # shared cls head
        logits = self.cls_head(feats[mask])
        logits_cls_pt = self.cls_head(feats_cls_pt[mask])

        loss_1 = self.loss_module(logits, target)
        loss_2 = self.loss_module(logits_cls_pt, target)
        return loss_1, loss_2
