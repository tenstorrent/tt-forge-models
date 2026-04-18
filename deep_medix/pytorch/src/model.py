# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
DeepMedix-R1 model wrapper for extracting logits from model outputs.

The Qwen2.5-VL vision encoder contains data-dependent operations
(get_window_index, torch.unique_consecutive) that produce incorrect tensor
shapes under torch.compile / XLA tracing. We bypass it by accepting
pre-computed inputs_embeds and position_ids instead.
"""

import torch


# https://github.com/tenstorrent/tt-xla/issues/1661
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds, attention_mask, position_ids):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return outputs.logits
