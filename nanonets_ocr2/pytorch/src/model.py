# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Nanonets OCR2 model wrapper that takes pre-computed embeddings and
only compiles the language model + lm_head for TT device.
"""

import torch


# https://github.com/tenstorrent/tt-xla/issues/1661
class Wrapper(torch.nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.language_model = full_model.model.language_model
        self.lm_head = full_model.lm_head

    def forward(self, inputs_embeds, position_ids, attention_mask):
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        return self.lm_head(outputs.last_hidden_state)
