# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Janus-Pro understanding image-encoder module.

Understanding path (reference ``janus`` ``inference.py`` / ``prepare_inputs_embeds``):
the SigLIP ``vision_model`` (CLIPVisionTower) encodes a 384x384 image into patch
features, which the MLP ``aligner`` projects into the language-model embedding space.
The result is the per-image token embeddings that get spliced into the text prompt.
"""

import torch
import torch.nn as nn


class JanusUnderstandImageEncoder(nn.Module):
    """SigLIP vision tower + MLP aligner -> image token embeddings.

    forward(pixel_values [B, 3, 384, 384]) -> image_embeds [B, 576, 4096].
    """

    def __init__(self, vision_model: nn.Module, aligner: nn.Module):
        super().__init__()
        self.vision_model = vision_model
        self.aligner = aligner

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.vision_model(pixel_values)
        return self.aligner(features)
