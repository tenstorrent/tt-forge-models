# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen 2.5 VL for Video Classification model wrapper for extracting logits from model outputs.
"""

import torch


@torch._dynamo.disable
def _process_vision(model, input_ids, pixel_values, image_grid_thw, attention_mask):
    inner_model = model.model
    inputs_embeds = inner_model.get_input_embeddings()(input_ids)
    if pixel_values is not None:
        image_embeds = inner_model.get_image_features(
            pixel_values, image_grid_thw, return_dict=True
        ).pooler_output
        image_embeds = torch.cat(image_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = inner_model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    position_ids = inner_model.compute_3d_position_ids(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        second_per_grid_ts=None,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
    )
    return inputs_embeds, position_ids


# https://github.com/tenstorrent/tt-xla/issues/1661
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self, input_ids, attention_mask, pixel_values, image_grid_thw, **kwargs
    ):
        inputs_embeds, position_ids = _process_vision(
            self.model, input_ids, pixel_values, image_grid_thw, attention_mask
        )
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return outputs.logits
