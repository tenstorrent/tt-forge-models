# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Janus-Pro multimodal-understanding component modules (reference inference.py)."""

import torch
import torch.nn as nn


class JanusUnderstandPrefill(nn.Module):
    """Understanding prefill: language_model.model + lm_head over multimodal embeds.

    Takes the combined image+text ``inputs_embeds`` (built on CPU by the SigLIP
    vision tower + aligner + text embedding) and returns next-token text logits
    over the language vocab. This is the compute-dominant LLaMA-7B forward pass.
    """

    def __init__(self, mmgpt):
        super().__init__()
        self.mmgpt = mmgpt

    def forward(self, inputs_embeds):
        # Mirror the working generation ImageTokenStep prefill: use_cache=True with
        # a fresh cache. use_cache=False routes LLaMA through a mask path that emits
        # a cumsum on a uint8 tensor, which has no device kernel on blackhole.
        outputs = self.mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=None,
        )
        return self.mmgpt.language_model.lm_head(outputs.last_hidden_state[:, -1, :])


class JanusVisionEmbed(nn.Module):
    """SigLIP vision tower + understanding aligner: pixel_values -> image embeds."""

    def __init__(self, vision_model, aligner):
        super().__init__()
        self.vision_model = vision_model
        self.aligner = aligner

    def forward(self, pixel_values):
        return self.aligner(self.vision_model(pixel_values))
