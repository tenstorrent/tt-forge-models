# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch


class Wrapper(torch.nn.Module):
    """Wrapper that bakes image_grid_thw into the model state.

    In compile-only mode, integer input tensors get zero-initialized on the
    XLA device. The vision encoder's control flow depends on grid_thw values,
    so zero values cause cascading shape mismatches. By storing grid_thw as a
    buffer, it travels with the model weights and preserves its values.
    """

    def __init__(self, model, image_grid_thw):
        super().__init__()
        self.model = model
        self.register_buffer("image_grid_thw", image_grid_thw)

    def forward(self, input_ids, attention_mask, pixel_values):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": self.image_grid_thw,
        }
        outputs = self.model(**inputs)
        return outputs.logits
