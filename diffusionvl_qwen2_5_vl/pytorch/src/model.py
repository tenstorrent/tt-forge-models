# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
DiffusionVL Qwen 2.5 VL model wrapper for extracting logits from model outputs.
"""

import torch


# https://github.com/tenstorrent/tt-xla/issues/1661
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        # The model passes attention_mask straight to F.scaled_dot_product_attention,
        # which requires bool or float dtype, not the int64 that the processor returns.
        # Reshape [B, S] → [B, 1, 1, S] bool so it broadcasts correctly over heads.
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].bool()
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        outputs = self.model(**inputs)
        return outputs.logits
