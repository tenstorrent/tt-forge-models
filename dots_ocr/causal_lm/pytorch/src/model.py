# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wrapper that runs dots.ocr text-only (no image tokens), exercising just the
Qwen2 text decoder path, and returns logits.
"""
import torch


class TextDecoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        # No pixel_values / image tokens -> prepare_inputs_embeds returns plain
        # token embeddings and the Qwen2 decoder runs end to end.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.logits
