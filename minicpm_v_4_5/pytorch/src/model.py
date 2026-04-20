# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

_DATA_KEYS = frozenset(
    ["input_ids", "pixel_values", "tgt_sizes", "image_bound", "temporal_ids", "position_ids"]
)


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        data = {k: v for k, v in kwargs.items() if k in _DATA_KEYS}
        extra = {k: v for k, v in kwargs.items() if k not in _DATA_KEYS}
        outputs = self.model.forward(data, **extra)
        return outputs.logits
