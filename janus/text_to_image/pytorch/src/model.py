# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Path A step-0 module for Janus-Pro T2I (generation_inference.py loop i=0)."""

from __future__ import annotations

import torch.nn as nn
from janus.models import MultiModalityCausalLM


class JanusGitImageTokenStep0(nn.Module):
    """
    First image-token step: language_model.model + gen_head.

    Matches deepseek-ai/Janus generation_inference.py when i=0 and past_key_values is None.
    Output shape: [parallel_size * 2, image_token_vocab] (pre-CFG logits).
    """

    def __init__(self, mmgpt: MultiModalityCausalLM):
        super().__init__()
        self.mmgpt = mmgpt

    def forward(self, inputs_embeds):
        outputs = self.mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=None,
        )
        return self.mmgpt.gen_head(outputs.last_hidden_state[:, -1, :])
