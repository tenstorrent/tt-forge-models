# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch


class MedMOWrapper(torch.nn.Module):
    """Wraps MedMO (Qwen3VL) to run only the language model decoder + lm_head.

    The vision encoder uses dynamic control flow and ops (repeat, item) that
    are incompatible with torch.compile + TT backend.  Vision features are
    pre-computed on CPU in ModelLoader.load_inputs instead.
    """

    def __init__(self, model):
        super().__init__()
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head

    def forward(self, inputs_embeds, attention_mask, position_ids):
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = outputs[0]
        return self.lm_head(hidden_states)
