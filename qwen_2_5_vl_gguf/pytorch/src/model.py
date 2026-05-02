# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen 2.5 VL GGUF model wrapper for extracting logits from model outputs.
"""

import torch
import torch.nn.functional as F


def _patch_qwen2_5_vl_vision_forward():
    """Replace cu_seqlens computation in visual encoder with a CPU-based one.

    torch.repeat_interleave on XLA device gives wrong results when computing
    cu_seqlens (off by spatial_merge_unit on TT silicon). Compute it from
    grid_thw.tolist() values instead, which are already fetched via D2H
    in rot_pos_emb and get_window_index.
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
        BaseModelOutputWithPooling,
    )
    from transformers.utils.generic import merge_with_config_defaults
    from transformers.utils.output_capturing import capture_outputs

    if getattr(Qwen2_5_VisionTransformerPretrainedModel, "_cu_seqlens_patched", False):
        return

    @merge_with_config_defaults
    @capture_outputs
    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ):
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # Compute cu_seqlens from Python values (grid_thw already fetched via
        # D2H in rot_pos_emb and get_window_index) to avoid XLA bfloat16
        # integer arithmetic rounding (2204 → 2208) on TT silicon.
        grid_thw_list = grid_thw.tolist()
        seqlens_cpu = [t * h * w for t, h, w in grid_thw_list]
        cu_seqlens = torch.tensor(
            seqlens_cpu, dtype=torch.int32, device=hidden_states.device
        ).cumsum(dim=0)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        merged_hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        merged_hidden_states = merged_hidden_states[reverse_indices, :]

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )

    Qwen2_5_VisionTransformerPretrainedModel.forward = patched_forward
    Qwen2_5_VisionTransformerPretrainedModel._cu_seqlens_patched = True


def _patch_qwen2_5_vl_get_image_features():
    """Patch split_sizes computation to use Python-side grid_thw values.

    XLA bfloat16 integer arithmetic rounds 2204 → 2208 when computing
    image_grid_thw.prod(-1). Call .tolist() on the raw int64 tensor first
    so the arithmetic is done exactly in Python.
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLModel,
        BaseModelOutputWithPooling,
    )
    from transformers.utils.generic import can_return_tuple

    if getattr(Qwen2_5_VLModel, "_gif_patched", False):
        return

    @can_return_tuple
    def patched_get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_outputs = self.visual(
            pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs
        )
        # Compute split_sizes in Python from raw int64 grid_thw values to
        # avoid XLA bfloat16 arithmetic rounding on TT silicon.
        merge_size_sq = self.visual.spatial_merge_size ** 2
        split_sizes = [
            t * h * w // merge_size_sq
            for t, h, w in image_grid_thw.tolist()
        ]
        image_embeds = torch.split(vision_outputs.pooler_output, split_sizes)
        vision_outputs.pooler_output = image_embeds
        return vision_outputs

    Qwen2_5_VLModel.get_image_features = patched_get_image_features
    Qwen2_5_VLModel._gif_patched = True


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        _patch_qwen2_5_vl_vision_forward()
        _patch_qwen2_5_vl_get_image_features()
        if hasattr(model, "model") and hasattr(model.model, "visual"):
            model.model.visual.forward = torch.compiler.disable(
                model.model.visual.forward
            )

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        outputs = self.model(**inputs)
        return outputs.logits
