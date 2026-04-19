# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch


class Wrapper(torch.nn.Module):
    """Wrapper that bakes image_grid_thw into forward as a Python constant.

    In compile-only mode, integer tensors (both inputs and buffers) get
    zero-initialized during tracing. The vision encoder's control flow
    depends on grid_thw values, so zeros cause cascading shape mismatches.
    By storing the values as a Python list and recreating the tensor in
    forward, torch.compile embeds them as graph constants.
    """

    def __init__(self, model, image_grid_thw):
        super().__init__()
        self.model = model
        self._grid_thw_values = image_grid_thw.tolist()

    def forward(self, input_ids, attention_mask, pixel_values):
        image_grid_thw = torch.tensor(
            self._grid_thw_values,
            dtype=torch.int64,
            device=input_ids.device,
        )
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        outputs = self.model(**inputs)
        return outputs.logits
