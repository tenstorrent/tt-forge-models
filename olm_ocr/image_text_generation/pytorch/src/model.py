# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
OLM OCR (Qwen 2.5 VL based) model wrapper for extracting logits from model outputs.
"""

import types

import torch


def patch_visual_dtype(hf_model):
    """Replace get_image_features so it does not read ``self.visual.dtype``.

    The HF ``dtype`` property is ``next(p.dtype for p in self.parameters() ...)``,
    a generator over parameters that torch.compile/dynamo cannot trace (it raises
    ``NameError: ... free variable 'named_children'``). pixel_values is already
    fed in the vision tower's dtype, so we capture that dtype as a constant and
    cast against it, avoiding the un-traceable property entirely.
    """
    # Find the submodule that owns both `visual` and `get_image_features`
    # (Qwen2_5_VLModel), regardless of whether the top model delegates to `.model`.
    owner = None
    for module in hf_model.modules():
        if hasattr(module, "visual") and hasattr(module, "get_image_features"):
            owner = module
            break
    if owner is None:
        return hf_model

    visual_dtype = next(
        p.dtype for p in owner.visual.parameters() if p.is_floating_point()
    )

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        pixel_values = pixel_values.to(visual_dtype)
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)
        split_sizes = (
            image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        image_embeds = torch.split(vision_outputs.pooler_output, split_sizes)
        vision_outputs.pooler_output = image_embeds
        return vision_outputs

    owner.get_image_features = types.MethodType(get_image_features, owner)
    return hf_model


# Mirrors qwen_2_5_vl wrapper, see https://github.com/tenstorrent/tt-xla/issues/1661
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
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        }
        outputs = self.model(**inputs)
        return outputs.logits
