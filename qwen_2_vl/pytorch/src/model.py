# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen 2 VL model wrapper for extracting logits from model outputs.
"""

import torch


# https://github.com/tenstorrent/tt-xla/issues/1661
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds, attention_mask, position_ids):
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )
        return outputs.logits
