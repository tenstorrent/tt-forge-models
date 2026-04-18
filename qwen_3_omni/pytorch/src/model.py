# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen 3 Omni MoE model wrapper for extracting logits from model outputs.
"""

import torch


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, attention_mask, inputs_embeds):
        outputs = self.model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        return outputs.logits
