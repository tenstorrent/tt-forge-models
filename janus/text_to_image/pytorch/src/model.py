# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""First image-token step module for Janus-Pro text-to-image."""

from __future__ import annotations

import torch.nn as nn
from janus.models import MultiModalityCausalLM


class JanusGitImageTokenStep0(nn.Module):
    """
    First image-token step: language_model.model + gen_head.

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
