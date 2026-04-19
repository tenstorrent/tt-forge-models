# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch._dynamo


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch._dynamo.disable
    def _run_model(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        return self.model.model(**inputs)

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        outputs = self._run_model(input_ids, attention_mask, pixel_values, image_grid_thw)
        hidden_states = outputs.last_hidden_state
        logits = self.model.lm_head(hidden_states)
        return logits
