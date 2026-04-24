# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Ported from https://github.com/tenstorrent/tt-metal/blob/main/models/demos/unet_3d/torch_impl/

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=8):
        super(ConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        if in_channels == 1:
            num_groups = 1
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.norm.weight.data = torch.randn(in_channels)
        self.norm.bias.data = torch.randn(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        is_bottleneck,
        in_channels,
        hid_channels,
        out_channels,
        num_groups=8,
        kernel_size=3,
        scale_factor=2,
    ):
        super(Encoder, self).__init__()
        self.conv_block_1 = ConvBlock(
            in_channels, hid_channels, kernel_size, num_groups
        )
        self.conv_block_2 = ConvBlock(
            hid_channels, out_channels, kernel_size, num_groups
        )
        self.pool = nn.MaxPool3d(kernel_size=scale_factor, stride=scale_factor)
        self.is_bottleneck = is_bottleneck

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        if self.is_bottleneck:
            return x
        x_pooled = self.pool(x)
        return x_pooled, x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=8):
        super(Decoder, self).__init__()
        self.conv_block_1 = ConvBlock(
            in_channels, out_channels, kernel_size, num_groups
        )
        self.conv_block_2 = ConvBlock(
            out_channels, out_channels, kernel_size, num_groups
        )

    def forward(self, x, skip_connection):
        size_to_match = skip_connection.shape[2:]
        x = F.interpolate(x, size=size_to_match, mode="nearest")
        x = torch.cat((skip_connection, x), dim=1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=32,
        num_levels=3,
        num_groups=8,
        scale_factor=2,
    ):
        super(UNet3D, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        c = base_channels
        for level in range(num_levels):
            in_ch = in_channels if level == 0 else c
            self.encoders.append(
                Encoder(
                    is_bottleneck=False,
                    in_channels=in_ch,
                    hid_channels=c,
                    out_channels=c * 2,
                    num_groups=num_groups,
                    scale_factor=scale_factor,
                )
            )
            c *= 2

        self.bottleneck = Encoder(
            is_bottleneck=True,
            in_channels=c,
            hid_channels=c,
            out_channels=c * 2,
            num_groups=num_groups,
        )

        for level in range(num_levels):
            self.decoders.append(Decoder(c * 3, c, num_groups=num_groups))
            c //= 2

        self.final_conv = nn.Conv3d(c * 2, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)

        x = self.bottleneck(x)

        for decoder in self.decoders:
            skip = skip_connections.pop()
            x = decoder(x, skip)

        x = self.final_conv(x)
        x = torch.nn.functional.sigmoid(x)
        return x
