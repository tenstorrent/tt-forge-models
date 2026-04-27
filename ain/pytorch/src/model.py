# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
AIN model wrapper for extracting logits from model outputs.
"""

import copy
import torch


def _xla_int_to_cpu(tensor):
    """Transfer an integer tensor from XLA to CPU using float32 as intermediary.

    TT XLA cannot transfer int32/int64 tensors from device to host (error code 13).
    Float32 transfers succeed. For integer values < 2^24 = 16,777,216 (token IDs,
    grid dimensions, attention masks), int→float32 is lossless.
    """
    return tensor.float().cpu().long()


@torch.compiler.disable
def _precompute_inputs_on_cpu(
    cpu_embed,
    cpu_visual,
    get_rope_index_fn,
    image_token_id,
    input_ids,
    attention_mask,
    pixel_values,
    image_grid_thw,
):
    """Pre-compute inputs_embeds and position_ids entirely on CPU.

    TT XLA cannot handle:
    - int64 device→host transfers (error code 13) needed by rot_pos_emb, get_rope_index
    - dynamic-shape ops (inputs_embeds[bool_mask].numel()) in get_placeholder_mask
    - potentially int64 arithmetic (repeat_interleave) for cu_seqlens

    By pre-computing inputs_embeds (text+image merged) and position_ids on CPU,
    we pass only float32 tensors to the XLA model and skip all integer-dependent
    model paths (get_image_features, get_placeholder_mask, compute_3d_position_ids).
    """
    original_device = input_ids.device

    # Transfer tensors to CPU.  Float32 XLA→CPU works; int64 needs float32 bridge.
    if original_device.type != "cpu":
        input_ids_cpu = _xla_int_to_cpu(input_ids)
        attn_mask_cpu = (
            _xla_int_to_cpu(attention_mask) if attention_mask is not None else None
        )
        pixel_values_cpu = pixel_values.cpu()  # float32 XLA → CPU
        grid_thw_cpu = _xla_int_to_cpu(image_grid_thw)
    else:
        input_ids_cpu = input_ids.long()
        attn_mask_cpu = attention_mask.long() if attention_mask is not None else None
        pixel_values_cpu = pixel_values
        grid_thw_cpu = image_grid_thw.long()

    # 1. Text embeddings on CPU
    inputs_embeds = cpu_embed(input_ids_cpu)

    # 2. Visual embeddings on CPU using CPU visual encoder
    spatial_merge_size = cpu_visual.spatial_merge_size
    pixel_values_typed = pixel_values_cpu.type(cpu_visual.dtype)
    vision_outputs = cpu_visual(pixel_values_typed, grid_thw=grid_thw_cpu, return_dict=True)
    split_sizes = [
        int(s) for s in (grid_thw_cpu.prod(-1) // spatial_merge_size**2).tolist()
    ]
    image_embeds = torch.cat(
        torch.split(vision_outputs.pooler_output, split_sizes), dim=0
    ).to(inputs_embeds.dtype)

    # 3. Merge image embeddings into text embeddings (replace image token positions)
    special_image_mask = (input_ids_cpu == image_token_id).unsqueeze(-1).expand_as(
        inputs_embeds
    )
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_embeds)

    # 4. Compute 3D position IDs on CPU
    position_ids, _ = get_rope_index_fn(
        input_ids_cpu,
        image_grid_thw=grid_thw_cpu,
        attention_mask=attn_mask_cpu,
    )

    if original_device.type != "cpu":
        inputs_embeds = inputs_embeds.to(original_device)
        position_ids = position_ids.to(original_device)

    return inputs_embeds, position_ids


# https://github.com/tenstorrent/tt-xla/issues/1661
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._patch_cpu_assets()

    def _patch_cpu_assets(self):
        """Set up CPU-side assets used by _precompute_inputs_on_cpu.

        - visual encoder: remove from Qwen2VLModel._modules (so to('xla') skips it)
          and store as a plain __dict__ attribute so it stays on CPU.
        - embed_tokens: deep-copy to CPU, stored on Wrapper outside _modules.
        - get_rope_index: stored as a bound-method reference (called with CPU tensors).

        TT XLA cannot handle integer operations in the vision path (rot_pos_emb,
        cu_seqlens, split_sizes, get_placeholder_mask dynamic shapes). Running
        everything on CPU before the XLA forward avoids all of these.
        """
        qwen2vl_model = self.model.model  # Qwen2VLModel

        # CPU visual encoder: remove from _modules so Module.to('xla') won't move it
        visual_cpu = copy.deepcopy(qwen2vl_model.visual).cpu()
        qwen2vl_model._modules.pop("visual", None)
        object.__setattr__(qwen2vl_model, "visual", visual_cpu)

        # CPU embed_tokens: deep-copy so it stays on CPU regardless of model.to('xla')
        embed_tokens_cpu = copy.deepcopy(
            qwen2vl_model.language_model.embed_tokens
        ).cpu()
        object.__setattr__(self, "_cpu_embed", embed_tokens_cpu)

        # Reference to get_rope_index (bound to qwen2vl_model, runs on CPU tensors)
        object.__setattr__(self, "_get_rope_index", qwen2vl_model.get_rope_index)

        # image_token_id needed to build special_image_mask on CPU
        object.__setattr__(
            self,
            "_image_token_id",
            qwen2vl_model.config.image_token_id,
        )

    def forward(
        self, input_ids, attention_mask, pixel_values, image_grid_thw, **kwargs
    ):
        # Pre-compute inputs_embeds (text+image merged) and position_ids on CPU.
        # Returns float32 tensors on the same device as the inputs.
        inputs_embeds, position_ids = _precompute_inputs_on_cpu(
            self._cpu_embed,
            self.model.model.visual,  # CPU visual (from __dict__)
            self._get_rope_index,
            self._image_token_id,
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
        )

        # Pass pre-computed embeddings and positions to the XLA model.
        # Omit input_ids, pixel_values, image_grid_thw so the model skips all
        # integer-dependent paths (get_image_features, get_placeholder_mask,
        # compute_3d_position_ids).
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        return outputs.logits
