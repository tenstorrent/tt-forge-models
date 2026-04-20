# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import nn
from transformers.models.canine.modeling_canine import CanineSelfAttention


def patch_canine_attention(model):
    original_forward = CanineSelfAttention.forward

    def patched_forward(
        self, from_tensor, to_tensor, attention_mask=None, output_attentions=False
    ):
        batch_size, seq_length, _ = from_tensor.shape

        key_layer = (
            self.key(to_tensor)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        value_layer = (
            self.value(to_tensor)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        query_layer = (
            self.query(from_tensor)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            if attention_mask.ndim == 3:
                attention_mask = torch.unsqueeze(attention_mask, dim=1)
                attention_mask = (1.0 - attention_mask.float()) * torch.finfo(
                    attention_scores.dtype
                ).min
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.to(value_layer.dtype)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs

    CanineSelfAttention.forward = patched_forward
