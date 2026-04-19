# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import types

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling


def _safe_fast_pos_embed_interpolate(self, grid_thw):
    """Patched version that handles zero-valued grid_thw from compile-only mode."""
    grid_thw_list = grid_thw.tolist()
    grid_ts = [row[0] for row in grid_thw_list]
    grid_hs = [row[1] for row in grid_thw_list]
    grid_ws = [row[2] for row in grid_thw_list]

    if all(t == 0 and h == 0 and w == 0 for t, h, w in grid_thw_list):
        return torch.zeros(
            0,
            self.config.hidden_size,
            device=self.pos_embed.weight.device,
            dtype=self.pos_embed.weight.dtype,
        )

    device = self.pos_embed.weight.device
    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]

    for t, h, w in grid_thw_list:
        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * self.num_grid_per_side
        base_h_ceil = h_idxs_ceil * self.num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(
        weight_list, dtype=self.pos_embed.weight.dtype, device=device
    )
    pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

    patch_pos_embeds_permute = []
    merge_size = self.config.spatial_merge_size
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
        if t == 0:
            patch_pos_embeds_permute.append(
                torch.zeros(
                    0, pos_embed.shape[-1], device=device, dtype=pos_embed.dtype
                )
            )
            continue
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(
                t, h // merge_size, merge_size, w // merge_size, merge_size, -1
            )
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)

    if not patch_pos_embeds_permute:
        return torch.zeros(
            0, self.config.hidden_size, device=device, dtype=self.pos_embed.weight.dtype
        )
    patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
    return patch_pos_embeds


def _safe_vision_forward(self, hidden_states, grid_thw, **kwargs):
    """Patched vision forward that handles zero-valued grid_thw."""
    hidden_states = self.patch_embed(hidden_states)
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
    if pos_embeds.shape[0] != hidden_states.shape[0]:
        pos_embeds = torch.zeros_like(hidden_states)
    hidden_states = hidden_states + pos_embeds

    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    repeat_counts = grid_thw[:, 1] * grid_thw[:, 2]
    repeat_times = grid_thw[:, 0]
    if repeat_counts.sum() == 0:
        cu_seqlens = torch.zeros(1, dtype=torch.int32, device=hidden_states.device)
    else:
        cu_seqlens = torch.repeat_interleave(repeat_counts, repeat_times).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for blk in self.blocks:
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    merged_hidden_states = self.merger(hidden_states)

    return BaseModelOutputWithPooling(
        last_hidden_state=hidden_states,
        pooler_output=merged_hidden_states,
    )


def _patch_vision_model(model):
    """Patch the vision model to handle zero-valued grid_thw."""
    for module in model.modules():
        if type(module).__name__ == "Qwen3_5VisionModel":
            module.fast_pos_embed_interpolate = types.MethodType(
                _safe_fast_pos_embed_interpolate, module
            )
            # Also need to patch the forward since it has repeat_interleave
            # that fails with zero grid_thw
            module.forward = types.MethodType(_safe_vision_forward, module)
            break


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        _patch_vision_model(model)

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        outputs = self.model(**inputs)
        return outputs.logits
