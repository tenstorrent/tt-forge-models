# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen 2 VL model wrapper for extracting logits from model outputs.
"""

import torch


# https://github.com/tenstorrent/tt-xla/issues/1661
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head
        # Store full model outside nn.Module submodule tracking so dynamo
        # never discovers the vision encoder during compilation.
        self._eager_model = [model]

    @torch.compiler.disable
    def _prepare_inputs(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        model = self._eager_model[0]
        inputs_embeds = model.get_input_embeddings()(input_ids)

        pixel_values = pixel_values.type(model.model.visual.dtype)
        image_outputs = model.model.get_image_features(
            pixel_values, image_grid_thw, return_dict=True
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        image_mask, _ = model.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        position_ids = model.model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        return inputs_embeds, position_ids

    def forward(
        self, input_ids, attention_mask, pixel_values, image_grid_thw, **kwargs
    ):
        inputs_embeds, position_ids = self._prepare_inputs(
            input_ids, attention_mask, pixel_values, image_grid_thw
        )
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
        )
        return self.lm_head(outputs.last_hidden_state)
