# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch


@torch.compiler.disable
def _make_cpu_int_tensor(values, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype)


def _patch_vision_pipeline(model, saved_inputs):
    """Patch vision-dependent methods to use saved CPU integer tensors.

    The XLA device zeroes all integer tensors. The vision pipeline depends on
    correct integer values (grid_thw for shapes, input_ids for token matching).
    Each patched method runs outside torch.compile with CPU integer tensors
    containing the original values, while the language model is still compiled.
    """
    inner = model.model

    grid_thw_vals = saved_inputs["image_grid_thw"].tolist()
    input_ids_vals = saved_inputs["input_ids"].tolist()
    attn_mask_vals = saved_inputs["attention_mask"].tolist()

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

    original_c3d = inner.compute_3d_position_ids

    @torch.compiler.disable
    def patched_compute_3d_position_ids(
        input_ids=None, image_grid_thw=None, attention_mask=None, **kwargs
    ):
        target_device = input_ids.device if input_ids is not None else "cpu"
        cpu_ids = torch.tensor(input_ids_vals, dtype=torch.int64)
        cpu_grid_thw = (
            torch.tensor(grid_thw_vals, dtype=torch.int64)
            if image_grid_thw is not None
            else None
        )
        cpu_mask = torch.tensor(attn_mask_vals, dtype=torch.int64)
        result = original_c3d(
            input_ids=cpu_ids,
            image_grid_thw=cpu_grid_thw,
            attention_mask=cpu_mask,
            **kwargs,
        )
        if isinstance(result, tuple):
            return tuple(
                r.to(target_device) if isinstance(r, torch.Tensor) else r
                for r in result
            )
        return result

    inner.compute_3d_position_ids = patched_compute_3d_position_ids


class Wrapper(torch.nn.Module):
    def __init__(self, model, saved_inputs):
        super().__init__()
        self.model = model
        self._grid_thw_values = saved_inputs["image_grid_thw"].tolist()
        _patch_vision_pipeline(model, saved_inputs)

    def forward(self, input_ids, attention_mask, pixel_values):
        image_grid_thw = _make_cpu_int_tensor(self._grid_thw_values)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        outputs = self.model(**inputs)
        return outputs.logits
