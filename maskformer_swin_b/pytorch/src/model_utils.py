# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from typing import Optional


class MaskformerEncoderWrapper(torch.nn.Module):
    def __init__(self, model, num_layers: Optional[int] = None):
        super().__init__()
        self.model = model
        self.input_dimensions = (160, 216)
        default_num_layers = getattr(self.model, "num_layers", 4)
        self.head_mask = [None] * default_num_layers
        self.output_attentions = False
        self.output_hidden_states = True
        self.return_dict = True
        if num_layers is not None:
            self.set_num_layers(num_layers)

    def set_num_layers(self, num_layers: int):
        max_layers = len(getattr(self.model, "layers", [])) or getattr(
            self.model, "num_layers", num_layers
        )
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if max_layers is not None and num_layers > max_layers:
            raise ValueError(
                f"num_layers {num_layers} exceeds available layers {max_layers}"
            )
        if hasattr(self.model, "num_layers"):
            self.model.num_layers = num_layers
        if hasattr(self.model, "layers") and isinstance(
            self.model.layers, torch.nn.ModuleList
        ):
            self.model.layers = torch.nn.ModuleList(
                list(self.model.layers)[:num_layers]
            )
        self.head_mask = [None] * num_layers

    def forward(self, hidden_states):
        outputs = self.model(
            hidden_states=hidden_states,
            input_dimensions=self.input_dimensions,
            head_mask=self.head_mask,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
            return_dict=self.return_dict,
        )
        return (outputs.last_hidden_state, outputs.hidden_states)
