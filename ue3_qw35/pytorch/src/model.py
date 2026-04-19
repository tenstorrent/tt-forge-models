# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch


@torch.compiler.disable
def _make_cpu_grid_thw(values):
    return torch.tensor(values, dtype=torch.int64)


class Wrapper(torch.nn.Module):
    """Wrapper that keeps image_grid_thw on CPU to avoid XLA integer zeroing.

    The XLA device zeroes all integer tensors, even those created from Python
    constants. By creating image_grid_thw inside a torch.compiler.disable
    region, it stays on CPU with correct values. The vision encoder methods
    that consume it (fast_pos_embed_interpolate, rot_pos_emb, get_rope_index)
    all call .tolist() first, so the CPU location is fine.
    """

    def __init__(self, model, image_grid_thw):
        super().__init__()
        self.model = model
        self._grid_thw_values = image_grid_thw.tolist()

    def forward(self, input_ids, attention_mask, pixel_values):
        image_grid_thw = _make_cpu_grid_thw(self._grid_thw_values)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        outputs = self.model(**inputs)
        return outputs.logits
