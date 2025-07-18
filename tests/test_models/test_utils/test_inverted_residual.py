# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.models.utils import InvertedResidual, SELayer


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


def test_inverted_residual():

    with pytest.raises(AssertionError):
        # stride must be in [1, 2]
        InvertedResidual(16, 16, 32, stride=3)

    with pytest.raises(AssertionError):
        # se_cfg must be None or dict
        InvertedResidual(16, 16, 32, se_cfg=list())

    # Add expand conv if in_channels and mid_channels is not the same
    assert InvertedResidual(32, 16, 32).with_expand_conv is False
    assert InvertedResidual(16, 16, 32).with_expand_conv is True

    # Test InvertedResidual forward, stride=1
    block = InvertedResidual(16, 16, 32, stride=1)
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert getattr(block, 'se', None) is None
    assert block.with_res_shortcut
    assert x_out.shape == torch.Size((1, 16, 56, 56))

    # Test InvertedResidual forward, stride=2
    block = InvertedResidual(16, 16, 32, stride=2)
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert not block.with_res_shortcut
    assert x_out.shape == torch.Size((1, 16, 28, 28))

    # Test InvertedResidual forward with se layer
    se_cfg = dict(channels=32)
    block = InvertedResidual(16, 16, 32, stride=1, se_cfg=se_cfg)
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert isinstance(block.se, SELayer)
    assert x_out.shape == torch.Size((1, 16, 56, 56))

    # Test InvertedResidual forward without expand conv
    block = InvertedResidual(32, 16, 32)
    x = torch.randn(1, 32, 56, 56)
    x_out = block(x)
    assert getattr(block, 'expand_conv', None) is None
    assert x_out.shape == torch.Size((1, 16, 56, 56))

    # Test InvertedResidual forward with GroupNorm
    block = InvertedResidual(16,
                             16,
                             32,
                             norm_cfg=dict(type='GN', num_groups=2))
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    for m in block.modules():
        if is_norm(m):
            assert isinstance(m, GroupNorm)
    assert x_out.shape == torch.Size((1, 16, 56, 56))

    # Test InvertedResidual forward with HSigmoid
    block = InvertedResidual(16, 16, 32, act_cfg=dict(type='HSigmoid'))
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size((1, 16, 56, 56))

    # Test InvertedResidual forward with checkpoint
    block = InvertedResidual(16, 16, 32, with_cp=True)
    x = torch.randn(1, 16, 56, 56)
    x_out = block(x)
    assert block.with_cp
    assert x_out.shape == torch.Size((1, 16, 56, 56))
