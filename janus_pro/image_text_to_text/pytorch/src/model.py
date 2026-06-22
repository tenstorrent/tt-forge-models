# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Janus-Pro multimodal-understanding component modules.

The understanding pathway (reference ``Janus`` ``MultiModalityCausalLM.
prepare_inputs_embeds``) runs the SigLIP vision tower followed by the
understanding MLP aligner to turn raw pixels into language-model embeddings,
which are then spliced into the text token stream before the LLM prefill.

``JanusUndVision`` isolates that vision pathway (``aligner(vision_model(x))``)
as a single compilable forward. This is the understanding-specific half of
Janus-Pro that the text-to-image (generation) loader does not cover.
"""

import torch
import torch.nn as nn


class JanusUndVision(nn.Module):
    """Understanding vision pathway: SigLIP vision tower + understanding aligner.

    forward:
        pixel_values [b * n_images, 3, 384, 384] (normalized)
        -> vision_model -> [b * n_images, 576, 1024]
        -> aligner      -> [b * n_images, 576, 4096]  (language-model embeds)
    """

    def __init__(self, vision_model: nn.Module, aligner: nn.Module):
        super().__init__()
        self.vision_model = vision_model
        self.aligner = aligner

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.aligner(self.vision_model(pixel_values))
