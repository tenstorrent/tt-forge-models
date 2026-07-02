# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
dots.ocr wrapper: run the full DotsOCRForCausalLM forward (vision tower +
Qwen2 text decoder) and return the language-model logits as a single tensor so
the model can be traced/compiled as one graph.
"""

import torch


class DotsOCRWrapper(torch.nn.Module):
    """Adapts DotsOCRForCausalLM to a single-tensor forward for tracing.

    The HF forward accepts ``input_ids`` / ``pixel_values`` / ``image_grid_thw``
    and merges the vision embeddings into the text-token embeddings before
    running the Qwen2 decoder. The processor also emits ``mm_token_type_ids``,
    which the model forward does not consume, so it is intentionally dropped.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
        )
        return outputs.logits
