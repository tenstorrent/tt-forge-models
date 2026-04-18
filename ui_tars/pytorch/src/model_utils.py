# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch


def precompute_vision_inputs(model, inputs):
    """Run vision processing on CPU to precompute inputs_embeds and position_ids.

    The Qwen2VL vision pipeline contains many data-dependent operations
    (Python for-loops over grid_thw, .tolist(), boolean masking) that
    dynamo cannot trace with fake tensors. This runs the full vision
    preprocessing on CPU and returns inputs_embeds + position_ids that
    can be passed directly to the language model through XLA.
    """
    inner = model.model

    with torch.no_grad():
        inputs_embeds = inner.get_input_embeddings()(inputs["input_ids"])

        pixel_values = inputs["pixel_values"].type(inner.visual.get_dtype())
        image_grid_thw = inputs["image_grid_thw"]

        vision_outputs = inner.visual(
            pixel_values, grid_thw=image_grid_thw, return_dict=True
        )
        spatial_merge_size = inner.visual.spatial_merge_size
        split_sizes = (
            image_grid_thw.prod(-1) // spatial_merge_size**2
        ).tolist()
        image_embeds = torch.split(vision_outputs.pooler_output, split_sizes)
        image_embeds = torch.cat(image_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )

        image_mask, _ = inner.get_placeholder_mask(
            inputs["input_ids"],
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        position_ids = inner.compute_3d_position_ids(
            input_ids=inputs["input_ids"],
            image_grid_thw=image_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=inputs["attention_mask"],
        )

    return inputs_embeds, position_ids
