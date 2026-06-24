# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Molmo2 model wrapper.

Wraps the HuggingFace ``Molmo2ForConditionalGeneration`` so the forward pass
takes an explicit, fixed set of multimodal tensors and returns the logits
tensor directly. This keeps the traced graph free of the Python-side output
dataclass and of the (unused at inference) KV-cache machinery.
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
        image_token_pooling,
        image_grids,
        image_num_crops,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
            use_cache=False,
        )
        return outputs.logits
