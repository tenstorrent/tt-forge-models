# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ConvMixer architecture — reimplemented from the original source:
  https://github.com/locuslab/convmixer  (Patches Are All You Need?, Trockman & Kolter 2022)

The original GitHub code defines ConvMixer as a lambda / nn.Sequential factory.
This module reimplements the same architecture as explicit nn.Module classes so that:
  - Layers have stable, named parameters (required for weight-loading and debugging)
  - Padding uses explicit integers (XLA-friendly — avoids padding="same")
  - The module is importable and inspectable without executing the factory lambda
"""

import torch.nn as nn


class DepthwiseMixerBlock(nn.Module):
    """Depthwise conv + GELU + BN block with a residual connection.

    Corresponds to the inner Residual(...) in the original GitHub code.
    """

    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        # padding = kernel_size // 2 replicates padding="same" for odd kernels with stride=1
        self.depthwise = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, groups=dim, padding=kernel_size // 2
        )
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        return x + self.norm(self.act(self.depthwise(x)))


class PointwiseMixerBlock(nn.Module):
    """Pointwise (1×1) conv + GELU + BN block.

    Corresponds to the outer nn.Sequential layer in the original GitHub code.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.norm(self.act(self.pointwise(x)))


class ConvMixerLayer(nn.Module):
    """One ConvMixer stage: depthwise mixer (with residual) → pointwise mixer."""

    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.depthwise = DepthwiseMixerBlock(dim, kernel_size)
        self.pointwise = PointwiseMixerBlock(dim)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ConvMixer(nn.Module):
    """ConvMixer image classifier.

    Reference: https://github.com/locuslab/convmixer
    "Patches Are All You Need?" — Trockman & Kolter, 2022.

    Args:
        dim:         Hidden channel dimension (h in the paper).
        depth:       Number of ConvMixer layers.
        kernel_size: Depthwise conv kernel size (default 5).
        patch_size:  Patch-embedding stride (default 8).
        num_classes: Number of output classes (default 1000).
    """

    def __init__(
        self,
        dim: int = 256,
        depth: int = 8,
        kernel_size: int = 5,
        patch_size: int = 8,
        num_classes: int = 1000,
    ):
        super().__init__()

        # Patch embedding: non-overlapping patch projection
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

        # ConvMixer blocks
        self.layers = nn.Sequential(
            *[ConvMixerLayer(dim, kernel_size) for _ in range(depth)]
        )

        # Classifier head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.layers(x)
        return self.head(x)
