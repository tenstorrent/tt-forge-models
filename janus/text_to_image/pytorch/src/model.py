# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Janus-Pro T2I compile subgraph (not `JanusForConditionalGeneration.forward`).

Image generation in HF does not use `.forward()` for tokens. `generate(...,
generation_mode="image")` loops:
  language_model -> generation_head -> CFG logits_processor -> sample
  -> prepare_embeddings_for_image_generation -> next step (with KV cache).

This module matches **one iteration** of that loop (i=0), **before** CFG and sampling:
  same ops as modeling_janus.py lines ~1240-1251, with use_cache=False for a
  single full-sequence forward (bring-up subgraph; steps 1..575 use cache).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import JanusForConditionalGeneration


class JanusImageTokenLogitsStep(nn.Module):
    """LM forward on CFG prompt embeds -> generation_head logits (compile target)."""

    def __init__(self, janus: JanusForConditionalGeneration):
        super().__init__()
        self.janus_model = janus.model

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.janus_model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden = outputs.last_hidden_state[:, -1, :]
        return self.janus_model.generation_head(hidden)
