# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch


@torch.compiler.disable
def _make_cpu_grid_thw(values):
    return torch.tensor(values, dtype=torch.int64)


def _patch_vision_encoder(model):
    """Disable torch.compile on the vision encoder.

    The XLA device zeroes all integer tensors during compilation. The vision
    encoder depends heavily on integer grid_thw values for control flow and
    shape calculations (pos_embed, rotary_emb, cu_seqlens, attention splits).
    """
    visual = model.model.visual
    original_forward = visual.forward

    @torch.compiler.disable
    def disabled_forward(*args, **kwargs):
        return original_forward(*args, **kwargs)

    visual.forward = disabled_forward


class Wrapper(torch.nn.Module):
    def __init__(self, model, image_grid_thw):
        super().__init__()
        self.model = model
        self._grid_thw_values = image_grid_thw.tolist()
        _patch_vision_encoder(model)

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
