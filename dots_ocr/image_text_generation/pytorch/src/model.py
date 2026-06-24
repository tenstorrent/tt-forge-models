# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
dots.ocr model wrapper for extracting logits from model outputs.

The HuggingFace ``DotsOCRForCausalLM.forward`` returns a ``CausalLMOutputWithPast``
dataclass; the TT inference path needs a single tensor output and positional
inputs, so this thin wrapper maps the processor's keyword inputs onto the model
and returns just the logits.
"""

import torch


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
        )
        return outputs.logits
