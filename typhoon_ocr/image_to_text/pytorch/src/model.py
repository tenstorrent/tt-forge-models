# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Typhoon OCR model wrapper that bypasses the visual encoder on XLA.
"""

import torch


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds, attention_mask, position_ids):
        outputs = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=None,
        )
        return outputs.logits
