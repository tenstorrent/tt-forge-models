# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch


def _dynamo_safe_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """MoE experts forward rewritten to avoid data-dependent control flow
    (.cpu().numpy().tolist() and dynamic slicing) that breaks torch dynamo tracing."""
    tokens, hidden_dim = hidden_states.shape

    expert_outputs = torch.stack(
        [expert(hidden_states) for expert in self], dim=1
    )

    expanded_idx = top_k_index.unsqueeze(-1).expand(-1, -1, hidden_dim)
    selected = torch.gather(expert_outputs, 1, expanded_idx)

    result = (selected * top_k_weights.unsqueeze(-1)).sum(dim=1)
    return result.to(hidden_states.dtype)


def patch_moe_experts(model):
    for module in model.modules():
        cls = type(module)
        if cls.__name__ == "SarvamMoEExperts":
            cls.forward = _dynamo_safe_experts_forward
            break
