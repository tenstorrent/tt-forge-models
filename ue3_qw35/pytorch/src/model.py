# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import types

import torch


def _patch_vision_encoder(model):
    """Disable torch.compile on the vision encoder.

    The XLA device zeroes all integer tensors during compilation. The vision
    encoder depends heavily on integer grid_thw values for control flow and
    shape calculations (pos_embed, rotary_emb, cu_seqlens, attention splits).
    Disabling compilation on the vision encoder lets these run in eager mode
    with correct values while the text model is still compiled.
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
