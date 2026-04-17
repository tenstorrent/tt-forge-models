# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen 2 VL model wrapper for extracting logits from model outputs.
"""

import copy

import torch


# https://github.com/tenstorrent/tt-xla/issues/1661
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        object.__setattr__(
            self,
            "_cpu_visual",
            copy.deepcopy(model.model.visual).cpu().float().eval(),
        )
        object.__setattr__(self, "_cached_inputs_embeds", None)
        object.__setattr__(self, "_cached_position_ids", None)

    @torch.compiler.disable
    def _prepare_inputs(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        mm_token_type_ids=None,
    ):
        device = input_ids.device

        if self._cached_inputs_embeds is not None:
            inputs_embeds = self._cached_inputs_embeds.to(device)
            position_ids = (
                self._cached_position_ids.to(device)
                if self._cached_position_ids is not None
                else None
            )
            return inputs_embeds, position_ids

        inner = self.model.model
        inputs_embeds = inner.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            cpu_pixel_values = pixel_values.detach().cpu().float()
            cpu_grid_thw = image_grid_thw.detach().cpu()
            cpu_visual = self._cpu_visual
            with torch.no_grad():
                vision_outputs = cpu_visual(cpu_pixel_values, grid_thw=cpu_grid_thw)
            image_embeds = vision_outputs[1]
            split_sizes = (
                cpu_grid_thw.prod(-1) // cpu_visual.spatial_merge_size**2
            ).tolist()
            image_embeds = torch.split(image_embeds, split_sizes)
            image_embeds = torch.cat(image_embeds, dim=0).to(
                device, inputs_embeds.dtype
            )
            image_mask, _ = inner.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        position_ids = inner.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_type_ids,
        )

        object.__setattr__(self, "_cached_inputs_embeds", inputs_embeds.detach().cpu())
        if position_ids is not None:
            object.__setattr__(
                self, "_cached_position_ids", position_ids.detach().cpu()
            )

        return inputs_embeds, position_ids

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        mm_token_type_ids=None,
        **kwargs,
    ):
        inputs_embeds, position_ids = self._prepare_inputs(
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return outputs.logits
