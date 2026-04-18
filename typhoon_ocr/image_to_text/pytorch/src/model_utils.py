# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch


def precompute_inputs_embeds(model, processor_inputs):
    inner = model.model
    input_ids = processor_inputs["input_ids"]
    attention_mask = processor_inputs["attention_mask"]
    pixel_values = processor_inputs["pixel_values"]
    image_grid_thw = processor_inputs["image_grid_thw"]

    with torch.no_grad():
        inputs_embeds = inner.get_input_embeddings()(input_ids)

        image_outputs = inner.get_image_features(
            pixel_values, image_grid_thw, return_dict=True
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = inner.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        position_ids = inner.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
        )

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
