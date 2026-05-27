# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Janus-Pro T2I component modules (reference generation_inference.py)."""

import torch
import torch.nn as nn


class JanusGitImageTokenStep(nn.Module):
    """
    AR loop: language_model.model + gen_head.

    Step i=0: past_key_values=None, full CFG prompt embeds.
    Step i>=1: past_key_values from previous step, single-token embeds.

    Returns pre-CFG logits [parallel_size * 2, image_token_vocab] and updated KV cache.
    """

    def __init__(self, mmgpt):
        super().__init__()
        self.mmgpt = mmgpt

    def forward(self, inputs_embeds, past_key_values=None):
        from .model_utils import align_kv_cache_device

        past_key_values = align_kv_cache_device(past_key_values, inputs_embeds.device)
        outputs = self.mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values,
        )
        logits = self.mmgpt.gen_head(outputs.last_hidden_state[:, -1, :])
        return logits, outputs.past_key_values


class JanusGenImgEmbed(nn.Module):
    """gen_embed + gen_aligner (prepare_gen_img_embeds)."""

    def __init__(self, gen_embed, gen_aligner):
        super().__init__()
        self.gen_embed = gen_embed
        self.gen_aligner = gen_aligner

    def forward(self, image_ids):
        return self.gen_aligner(self.gen_embed(image_ids))


class JanusGenVisionDecode(nn.Module):
    """gen_vision_model.decode_code -> image tensor."""

    def __init__(self, gen_vision_model, decode_shape):
        super().__init__()
        self.gen_vision_model = gen_vision_model
        self.decode_shape = decode_shape

    def forward(self, generated_tokens):
        return self.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=self.decode_shape,
            channel_first=True,
        )
