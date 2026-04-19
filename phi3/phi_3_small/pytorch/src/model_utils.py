# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F


def _sdpa_attention(self, q, k, v, attention_mask=None, return_attention_probs=False):
    attn_output = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=True, scale=self.softmax_scale
    )
    if return_attention_probs:
        return attn_output, None
    return attn_output


def patch_attention(model):
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name == "Phi3SmallSelfAttention":
            module._apply_blocksparse_attention = _sdpa_attention.__get__(module)
            module._apply_dense_attention = _sdpa_attention.__get__(module)
    return model
