# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F
import torch._dynamo
from transformers.models.granitemoe.modeling_granitemoe import (
    GraniteMoeMoE,
    GraniteMoeParallelExperts,
)

_orig_parallel_forward = GraniteMoeParallelExperts.forward


def _safe_parallel_forward(self, inputs, expert_size):
    total = sum(expert_size)
    if total != inputs.size(0):
        n = inputs.size(0)
        base = n // self.num_experts
        remainder = n % self.num_experts
        expert_size = [
            base + (1 if i < remainder else 0) for i in range(self.num_experts)
        ]
    input_list = inputs.split(expert_size, dim=0)
    output_list = []
    for i in range(self.num_experts):
        output_list.append(F.linear(input_list[i], self.weight[i]))
    return torch.cat(output_list, dim=0)


GraniteMoeParallelExperts.forward = _safe_parallel_forward
GraniteMoeMoE.forward = torch._dynamo.disable(GraniteMoeMoE.forward)
