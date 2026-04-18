# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import torch._dynamo
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
)


def _patched_rot_pos_emb(self, grid_thw):
    grid_thw_list = grid_thw.cpu().tolist()
    merge_size = self.spatial_merge_size

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

        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(
            -1
        )
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(
            -1
        )

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
    Qwen2VisionTransformerPretrainedModel.rot_pos_emb = _patched_rot_pos_emb
    Qwen2VisionTransformerPretrainedModel.forward = torch._dynamo.disable(
        _patched_vision_forward
    )
    Qwen2VLModel.forward = torch._dynamo.disable(Qwen2VLModel.forward)
