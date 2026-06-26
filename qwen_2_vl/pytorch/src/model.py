# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen2-VL model wrapper for extracting logits from model outputs.

The runner passes positional tensors and expects a single tensor (logits)
back; this adapts the HF multi-input/ModelOutput interface to that.
"""

import torch


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        mm_token_type_ids=None,
    ):
        # Qwen2-VL requires mm_token_type_ids for multimodal RoPE (M-RoPE).
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        return outputs.logits
