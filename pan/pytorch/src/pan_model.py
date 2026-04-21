# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PAN (Pixel Attention Network) architecture for image super-resolution.

Based on the architecture from:
  "Efficient Image Super-Resolution Using Pixel Attention"
  (Zhao et al., ECCV 2020) — https://arxiv.org/abs/2010.01073
"""

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelAttentionBlock(nn.Module):
    """Pixel attention via a 1x1 convolution followed by sigmoid gating."""

    def __init__(self, nf):
        super().__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.sigmoid(self.conv(x))
        return torch.mul(x, y)


class PixelAttentionConv(nn.Module):
    """Pixel attention convolution block used inside SCPA."""

    def __init__(self, nf, k_size=3):
        super().__init__()
        self.k2 = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(
            nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.k4 = nn.Conv2d(
            nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )

    def forward(self, x):
        y = self.sigmoid(self.k2(x))
        out = torch.mul(self.k3(x), y)
        out = self.k4(out)
        return out


class SCPA(nn.Module):
    """Self-Calibrated Pixel Attention block (adapted from SCNet)."""

    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super().__init__()
        group_width = nf // reduction

        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)

        self.k1 = nn.Sequential(
            nn.Conv2d(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
        )
        self.PAConv = PixelAttentionConv(group_width)
        self.conv3 = nn.Conv2d(group_width * reduction, nf, kernel_size=1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x

        out_a = self.lrelu(self.conv1_a(x))
        out_b = self.lrelu(self.conv1_b(x))

        out_a = self.lrelu(self.k1(out_a))
        out_b = self.lrelu(self.PAConv(out_b))

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual
        return out


def _make_layer(block, n_layers):
    return nn.Sequential(*[block() for _ in range(n_layers)])


class PAN(nn.Module):
    """Pixel Attention Network for single-image super-resolution.

    Args:
        in_nc: Number of input channels.
        out_nc: Number of output channels.
        nf: Number of feature channels in the trunk.
        unf: Number of feature channels in the upsampling path.
        nb: Number of SCPA blocks in the trunk.
        scale: Upscaling factor (2, 3, or 4).
    """

    def __init__(self, in_nc=3, out_nc=3, nf=40, unf=24, nb=16, scale=2):
        super().__init__()
        self.scale = scale

        scpa_block = functools.partial(SCPA, nf=nf, reduction=2)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        self.SCPA_trunk = _make_layer(scpa_block, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PixelAttentionBlock(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PixelAttentionBlock(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.SCPA_trunk(fea))
        fea = fea + trunk

        if self.scale in (2, 3):
            fea = self.upconv1(
                F.interpolate(fea, scale_factor=self.scale, mode="nearest")
            )
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode="nearest"))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest"))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))

        out = self.conv_last(fea)
        ilr = F.interpolate(
            x, scale_factor=self.scale, mode="bilinear", align_corners=False
        )
        return out + ilr
