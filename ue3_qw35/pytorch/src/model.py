# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch


class Wrapper(torch.nn.Module):
    """Wrapper that keeps image_grid_thw on CPU to avoid XLA integer zeroing.

    The XLA device in compile-only mode zeroes all integer tensors, even those
    created from Python constants. Since the vision encoder only uses grid_thw
    via .tolist() for Python control flow, keeping it on CPU preserves the
    values while remaining functionally correct.
    """

    def __init__(self, model, image_grid_thw):
        super().__init__()
        self.model = model
        self._grid_thw_values = image_grid_thw.tolist()

    def forward(self, input_ids, attention_mask, pixel_values):
        image_grid_thw = torch.tensor(
            self._grid_thw_values,
            dtype=torch.int64,
        )
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        outputs = self.model(**inputs)
        return outputs.logits
