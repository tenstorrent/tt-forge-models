# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wrapper for the full dots.ocr pipeline that returns logits from a single
forward pass, so the compiled graph has a tensor (not a dataclass) output.
"""
import torch


class Wrapper(torch.nn.Module):
    """Run the full vision+decoder forward and expose only the logits."""

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
