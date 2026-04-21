# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal V-Express model architecture definitions for the VKpsGuider.
Adapted from: https://github.com/tencent-ailab/V-Express
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class InflatedConv3d(nn.Conv2d):
    """2D convolution applied per-frame over a 5D video tensor (b, c, f, h, w)."""

    def forward(self, x):
        video_length = x.shape[2]
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)
        return x


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class VKpsGuider(nn.Module):
    """V-KPS guider network that encodes facial keypoint images into a
    conditioning feature volume for the V-Express denoising UNet."""

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        super().__init__()
        self.conv_in = InflatedConv3d(
            conditioning_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1,
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                InflatedConv3d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )

        self.conv_out = zero_module(
            InflatedConv3d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)
        return embedding


# Default VKpsGuider configuration used by the V-Express inference pipeline.
V_KPS_GUIDER_PARAMS = {
    "conditioning_embedding_channels": 320,
    "block_out_channels": (16, 32, 96, 256),
}


def load_v_kps_guider(checkpoint_path, device="cpu"):
    """Instantiate the V-KPS guider and load its pretrained weights."""
    model = VKpsGuider(**V_KPS_GUIDER_PARAMS).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model
