# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch


@torch.compiler.disable
def _make_cpu_int_tensor(values, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype)


def _precompute_visual_pos_mask(saved_inputs, config):
    cpu_ids = saved_inputs["input_ids"]
    image_token_id = config.image_token_id
    return cpu_ids == image_token_id


def _patch_vision_pipeline(model, saved_inputs, saved_visual_pos_mask):
    inner = model.model

    grid_thw_vals = saved_inputs["image_grid_thw"].tolist()
    input_ids_vals = saved_inputs["input_ids"].tolist()

    visual = inner.visual
    original_visual_fwd = visual.forward

    @torch.compiler.disable
    def disabled_visual_fwd(*args, **kwargs):
        return original_visual_fwd(*args, **kwargs)

    visual.forward = disabled_visual_fwd

    original_gif = inner.get_image_features

    @torch.compiler.disable
    def patched_get_image_features(pixel_values, image_grid_thw, **kwargs):
        cpu_grid_thw = torch.tensor(grid_thw_vals, dtype=torch.int64)
        return original_gif(pixel_values, cpu_grid_thw, **kwargs)

    inner.get_image_features = patched_get_image_features

    @torch.compiler.disable
    def patched_get_placeholder_mask(input_ids, inputs_embeds=None, **kwargs):
        cpu_ids = torch.tensor(input_ids_vals, dtype=torch.int64)
        image_token_id = inner.config.image_token_id
        video_token_id = inner.config.video_token_id

        special_image_mask = cpu_ids == image_token_id
        special_video_mask = cpu_ids == video_token_id

        special_image_mask = (
            special_image_mask.unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        special_video_mask = (
            special_video_mask.unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        return special_image_mask, special_video_mask

    inner.get_placeholder_mask = patched_get_placeholder_mask

    lang_model = inner.language_model
    mask_indices = saved_visual_pos_mask.nonzero(as_tuple=False)

    @torch.compiler.disable
    def patched_deepstack_process(hidden_states, visual_pos_masks, visual_embeds):
        device = hidden_states.device
        dtype = hidden_states.dtype
        hs_cpu = hidden_states.detach().cpu().float()
        ve_cpu = visual_embeds.detach().cpu().float()
        hs_cpu = hs_cpu.clone()
        rows = mask_indices[:, 0]
        cols = mask_indices[:, 1]
        hs_cpu[rows, cols, :] = hs_cpu[rows, cols, :] + ve_cpu
        return hs_cpu.to(device=device, dtype=dtype)

    lang_model._deepstack_process = patched_deepstack_process


def _precompute_position_ids(model, saved_inputs):
    inner = model.model
    cpu_ids = saved_inputs["input_ids"]
    cpu_grid_thw = saved_inputs["image_grid_thw"]
    cpu_mask = saved_inputs["attention_mask"]

    with torch.no_grad():
        inputs_embeds = inner.get_input_embeddings()(cpu_ids)
        position_ids = inner.compute_3d_position_ids(
            input_ids=cpu_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=cpu_grid_thw,
            attention_mask=cpu_mask,
        )

    return position_ids.float()


class Wrapper(torch.nn.Module):
    def __init__(self, model, saved_inputs):
        super().__init__()
        self.model = model
        self._grid_thw_values = saved_inputs["image_grid_thw"].tolist()

        position_ids = _precompute_position_ids(model, saved_inputs)
        self.register_buffer("position_ids", position_ids)

        visual_pos_mask = _precompute_visual_pos_mask(
            saved_inputs, model.model.config
        )
        _patch_vision_pipeline(model, saved_inputs, visual_pos_mask)

    def forward(self, input_ids, attention_mask, pixel_values):
        image_grid_thw = _make_cpu_int_tensor(self._grid_thw_values)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "position_ids": self.position_ids,
        }
        outputs = self.model(**inputs)
        return outputs.logits
