# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling


# https://github.com/tenstorrent/tt-xla/issues/1661
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        mm_token_type_ids,
    ):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        }
        return self.model(**inputs)


def _patched_fast_pos_embed_interpolate(self, grid_thw):
    """Works around TT-XLA repeat/concatenate failing on zero or unit-sized dims."""
    grid_thw_list = grid_thw.tolist()
    device = self.pos_embed.weight.device

    valid = [(t, h, w) for t, h, w in grid_thw_list if h > 0 and w > 0 and t > 0]
    if not valid:
        embed_dim = self.pos_embed.weight.shape[1]
        return torch.zeros(
            0, embed_dim, device=device, dtype=self.pos_embed.weight.dtype
        )

    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]

    for t, h, w in valid:
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

    split_sizes = [h * w for _, h, w in valid]
    patch_pos_embeds = patch_pos_embeds.split(split_sizes)

    patch_pos_embeds_permute = []
    merge_size = self.config.spatial_merge_size
    for pos_embed, (t, h, w) in zip(patch_pos_embeds, valid):
        if t > 1:
            pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(
                t, h // merge_size, merge_size, w // merge_size, merge_size, -1
            )
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)

    return torch.cat(patch_pos_embeds_permute)


def _patched_rot_pos_emb(self, grid_thw):
    """Works around TT-XLA repeat failing on unit-sized dims in rot_pos_emb."""
    merge_size = self.spatial_merge_size
    grid_thw_list = grid_thw.tolist()

    valid = [(t, h, w) for t, h, w in grid_thw_list if h > 0 and w > 0 and t > 0]
    if not valid:
        dim = self.rotary_pos_emb.dim
        return torch.zeros(0, dim, device=grid_thw.device, dtype=torch.float32)

    max_hw = max(max(h, w) for _, h, w in valid)
    freq_table = self.rotary_pos_emb(max_hw)
    device = freq_table.device

    total_tokens = sum(t * h * w for t, h, w in valid)
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

    offset = 0
    for num_frames, height, width in valid:
        merged_h, merged_w = height // merge_size, width // merge_size

        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra_row = torch.arange(merge_size, device=device)
        intra_col = torch.arange(merge_size, device=device)

        row_idx = (
            block_rows[:, None, None, None] * merge_size
            + intra_row[None, None, :, None]
        )
        col_idx = (
            block_cols[None, :, None, None] * merge_size
            + intra_col[None, None, None, :]
        )

        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)

        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    embeddings = freq_table[pos_ids]
    embeddings = embeddings.flatten(1)
    return embeddings


def _patched_vision_forward(self, hidden_states, grid_thw, **kwargs):
    """Patched forward that handles zero-element tensors gracefully."""
    hidden_states = self.patch_embed(hidden_states)

    seq_len = hidden_states.shape[0]
    if seq_len == 0:
        hidden_dim = hidden_states.shape[1]
        empty = torch.zeros(
            0, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype
        )
        return BaseModelOutputWithPooling(
            last_hidden_state=empty,
            pooler_output=empty,
        )

    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds

    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    seq_len = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[1]
    hidden_states = hidden_states.reshape(seq_len, hidden_dim)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, rotary_pos_emb.shape[1])
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(
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


def apply_patches():
    import torch._dynamo
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5Model,
        Qwen3_5VisionModel,
    )

    Qwen3_5VisionModel.fast_pos_embed_interpolate = _patched_fast_pos_embed_interpolate
    Qwen3_5VisionModel.rot_pos_emb = _patched_rot_pos_emb
    Qwen3_5VisionModel.forward = torch._dynamo.disable(_patched_vision_forward)
    # Disable dynamo on the entire VL model forward to avoid:
    # - masked_scatter (unsupported by TT-MLIR)
    # - multiple graph breaks from .tolist() in get_image_features
    # - FakeTensor shape inference failures across graph breaks
    Qwen3_5Model.forward = torch._dynamo.disable(Qwen3_5Model.forward)
