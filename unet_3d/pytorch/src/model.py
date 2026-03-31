# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
3D UNet model for volumetric segmentation.

Reference: https://github.com/tenstorrent/Toyota (Toyota PR architecture)

GroupNorm-based U-Net architecture using 3D convolutions.
Used for medical image segmentation (e.g., brain tumor / organ segmentation).

Architecture:
  Encoder: GroupNorm + Conv3d + ReLU × 2 + stride-2 downsampling, repeated 3 times
  Bottleneck: GroupNorm + Conv3d + ReLU × 2
  Decoder: F.interpolate (nearest) + skip concat + ConvBlock × 2, × 3
  Output: Conv3d → sigmoid

Note on XLA/TT-MLIR compatibility:
  - MaxPool3d has limited bfloat16 support on TT-MLIR; replaced with stride-2 Conv3d.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """GroupNorm + Conv3d + ReLU block.

    Applies GroupNorm on input channels, then Conv3d, then ReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=8):
        super().__init__()
        padding = (kernel_size - 1) // 2
        if in_channels == 1:
            num_groups = 1
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    """Encoder block: two ConvBlocks + optional stride-2 downsampling.

    Uses a stride-2 Conv3d instead of MaxPool3d for XLA compatibility
    (MaxPool3d has limited bfloat16 support on TT-MLIR backends).
    """

    def __init__(self, is_bottleneck, in_channels, hid_channels, out_channels, num_groups=8, kernel_size=3, scale_factor=2):
        super().__init__()
        self.conv_block_1 = ConvBlock(in_channels, hid_channels, kernel_size, num_groups)
        self.conv_block_2 = ConvBlock(hid_channels, out_channels, kernel_size, num_groups)
        # Stride-2 Conv3d replaces MaxPool3d (bfloat16 workaround for TT-MLIR)
        self.pool = nn.Conv3d(out_channels, out_channels, kernel_size=scale_factor, stride=scale_factor, bias=False)
        self.is_bottleneck = is_bottleneck

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        if self.is_bottleneck:
            return x
        x_pooled = self.pool(x)
        return x_pooled, x


class Decoder(nn.Module):
    """Decoder block: nearest upsample + skip concat + two ConvBlocks."""

    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=8):
        super().__init__()
        self.conv_block_1 = ConvBlock(in_channels, out_channels, kernel_size, num_groups)
        self.conv_block_2 = ConvBlock(out_channels, out_channels, kernel_size, num_groups)

    def forward(self, x, skip_connection):
        size_to_match = skip_connection.shape[2:]
        x = F.interpolate(x, size=size_to_match, mode="nearest")
        x = torch.cat((skip_connection, x), dim=1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNet3D(nn.Module):
    """3D UNet for volumetric segmentation (Toyota PR architecture).

    3 encoder levels, 1 bottleneck, 3 decoders, final conv + sigmoid.
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=32, num_levels=3, num_groups=8, scale_factor=2):
        super().__init__()
        f = base_channels

        # Encoders
        self.encoder1 = Encoder(False, in_channels, f, f, num_groups=num_groups, scale_factor=scale_factor)
        self.encoder2 = Encoder(False, f, f * 2, f * 2, num_groups=num_groups, scale_factor=scale_factor)
        self.encoder3 = Encoder(False, f * 2, f * 4, f * 4, num_groups=num_groups, scale_factor=scale_factor)

        # Bottleneck
        self.bottleneck = Encoder(True, f * 4, f * 8, f * 8, num_groups=num_groups, scale_factor=scale_factor)

        # Decoders (in_channels = bottleneck_out + skip_out after concat)
        self.decoder3 = Decoder(f * 8 + f * 4, f * 4, num_groups=num_groups)
        self.decoder2 = Decoder(f * 4 + f * 2, f * 2, num_groups=num_groups)
        self.decoder1 = Decoder(f * 2 + f, f, num_groups=num_groups)

        # Output
        self.final_conv = nn.Conv3d(f, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, D, H, W) — volumetric input
        Returns:
            output: (B, out_channels, D, H, W) with sigmoid activation
        """
        x1_down, x1_skip = self.encoder1(x)
        x2_down, x2_skip = self.encoder2(x1_down)
        x3_down, x3_skip = self.encoder3(x2_down)

        bottleneck = self.bottleneck(x3_down)

        d3 = self.decoder3(bottleneck, x3_skip)
        d2 = self.decoder2(d3, x2_skip)
        d1 = self.decoder1(d2, x1_skip)

        out = self.final_conv(d1)
        out = self.sigmoid(out)
        return out
