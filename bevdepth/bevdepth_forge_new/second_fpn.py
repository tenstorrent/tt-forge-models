# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) Megvii Inc. All rights reserved.
import numpy as np
import torch
from torch import nn as nn
from third_party.tt_forge_models.bevdepth.bevdepth_forge_new.backbone import (
    build_norm_hardcoded,
)
from third_party.tt_forge_models.bevdepth.bevdepth_forge_new.common_imports import (
    BaseModule,
)

if torch.__version__ == "parrots":
    TORCH_VERSION = torch.__version__
else:
    TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])


def obsolete_torch_version(torch_version, version_threshold) -> bool:
    return torch_version == "parrots" or torch_version <= version_threshold


def build_upsample_layer(cfg, *args, **kwargs) -> nn.Module:
    """Minimal local builder for upsample layers.

    Supports:
    - type 'deconv': maps to nn.ConvTranspose2d, expects in_channels, out_channels, kernel_size, stride
    - type 'nearest' or 'bilinear': maps to nn.Upsample with mode
    """
    if cfg is None:
        cfg_ = dict(type="deconv", bias=False)
    else:
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict")
        if "type" not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")

    if layer_type == "deconv":
        # Expected kwargs: in_channels, out_channels, kernel_size, stride
        in_channels = kwargs.pop("in_channels")
        out_channels = kwargs.pop("out_channels")
        kernel_size = kwargs.pop("kernel_size")
        stride = kwargs.pop("stride")
        bias = cfg_.pop("bias", False)
        return nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias
        )
    elif layer_type in ("nearest", "bilinear"):
        return nn.Upsample(mode=layer_type, **cfg_)


def build_conv_layer(cfg, *args, **kwargs) -> nn.Module:
    """Minimal local builder for conv layers.

    Supports:
    - type 'Conv2d': maps to nn.Conv2d, expects in_channels, out_channels, kernel_size, stride
    """
    if cfg is None:
        cfg_ = dict(type="Conv2d", bias=False)
    else:
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict")
        if "type" not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")

    if layer_type == "Conv2d":
        in_channels = kwargs.pop("in_channels")
        out_channels = kwargs.pop("out_channels")
        kernel_size = kwargs.pop("kernel_size")
        stride = kwargs.pop("stride", 1)
        bias = cfg_.pop("bias", False)
        # Optional parameters with sensible defaults
        padding = cfg_.pop("padding", 0)
        dilation = cfg_.pop("dilation", 1)
        groups = cfg_.pop("groups", 1)
        padding_mode = cfg_.pop("padding_mode", "zeros")
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
    else:
        raise KeyError(f"Unsupported conv layer type: {layer_type}")


class NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, new_shape: tuple) -> torch.Tensor:
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple:
        shape = ctx.shape
        return NewEmptyTensorOp.apply(grad, shape), None


class ConvTranspose2d(nn.ConvTranspose2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 4)):
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d, op in zip(
                x.shape[-2:],
                self.kernel_size,
                self.padding,
                self.stride,
                self.dilation,
                self.output_padding,
            ):
                out_shape.append((i - 1) * s - 2 * p + (d * (k - 1) + 1) + op)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                # produce dummy gradient to avoid DDP warning.
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)


class SECONDFPN(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(
        self,
        in_channels=[128, 128, 256],
        out_channels=[256, 256, 256],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        conv_cfg=dict(type="Conv2d", bias=False),
        use_conv_for_no_stride=False,
        init_cfg=None,
    ):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i],
                )
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride,
                )

            name, norm = build_norm_hardcoded(norm_cfg, out_channel)
            deblock = nn.Sequential(upsample_layer, norm, nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

        if init_cfg is None:
            self.init_cfg = [
                dict(type="Kaiming", layer="ConvTranspose2d"),
                dict(type="Constant", layer="NaiveSyncBatchNorm2d", val=1.0),
            ]

    # @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return [out]
