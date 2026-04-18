# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


class SuperPointEncoderWrapper(nn.Module):
    """Wraps SuperPointForKeypointDetection to return only fixed-size encoder output.

    The full model post-processes encoder features into variable-length keypoints
    via NMS, which produces dynamic output shapes incompatible with compilation.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        pixel_values = self.model.extract_one_channel_pixel_values(pixel_values)
        encoder_outputs = self.model.encoder(pixel_values, return_dict=False)
        return encoder_outputs[0]
