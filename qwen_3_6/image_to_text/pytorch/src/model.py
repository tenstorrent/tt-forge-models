# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen 3.6 VL model wrapper.

Wraps ``Qwen3_5ForConditionalGeneration`` so it can be traced/compiled by the
TT backend:

* Pins the vision tower dtype to a concrete value. The stock
  ``get_image_features`` casts ``pixel_values`` with ``self.visual.dtype``,
  whose property body iterates a ``self.parameters()`` generator. TorchDynamo
  cannot trace that generator (it raises
  ``InternalTorchDynamoError: ... free variable 'named_children'``), so we
  replace ``get_image_features`` with an equivalent that uses a dtype captured
  eagerly at wrap time.
* Exposes an explicit, positional forward signature and returns only the
  ``logits`` tensor, so the runner's pytree comparator diffs a single tensor
  rather than a model output dataclass.
"""

import types

import torch


def _make_patched_get_image_features(visual_dtype):
    """Build a ``get_image_features`` that avoids the ``self.visual.dtype``
    generator-property access that breaks TorchDynamo."""

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        # The caller passes return_dict=True; drop it from kwargs so it does not
        # collide with the explicit return_dict below (the stock method strips
        # it via the @can_return_tuple decorator, which we don't reuse here).
        kwargs.pop("return_dict", None)
        pixel_values = pixel_values.to(visual_dtype)
        vision_output = self.visual(
            pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs
        )
        image_embeds = vision_output.pooler_output
        split_sizes = (
            image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
        ).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        vision_output.pooler_output = image_embeds
        return vision_output

    return get_image_features


class Wrapper(torch.nn.Module):
    """Trace-friendly wrapper around Qwen3_5ForConditionalGeneration."""

    def __init__(self, model):
        super().__init__()
        self.model = model

        # Capture the vision tower's floating-point dtype eagerly (outside any
        # traced region) and patch get_image_features to use it.
        inner = model.model
        visual_dtype = next(
            p.dtype for p in inner.visual.parameters() if p.is_floating_point()
        )
        inner.get_image_features = types.MethodType(
            _make_patched_get_image_features(visual_dtype), inner
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        mm_token_type_ids=None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        return outputs.logits
