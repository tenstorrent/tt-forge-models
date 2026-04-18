# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import transformers.integrations.moe as moe_module

_grouped_linear = moe_module._grouped_linear


def _patched_grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    token_idx = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1)
        .expand(-1, num_top_k)
        .reshape(-1)
    )
    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)

    selected_hidden_states = hidden_states[token_idx]

    perm = torch.argsort(expert_ids)
    inv_perm = torch.argsort(perm)
    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    selected_hidden_states_g = selected_hidden_states[perm]

    selected_gate_up = self.gate_up_proj
    selected_down = self.down_proj
    selected_gate_up_bias = (
        self.gate_up_proj_bias[expert_ids_g] if self.has_bias else None
    )
    selected_down_bias = (
        self.down_proj_bias[expert_ids_g] if self.has_bias else None
    )

    histc_input = expert_ids_g.float()
    num_tokens_per_expert = torch.histc(
        histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1
    )
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    gate_up_out = _grouped_linear(
        selected_hidden_states_g,
        selected_gate_up,
        selected_gate_up_bias,
        offsets,
        is_transposed=self.is_transposed,
    )

    gated_out = self._apply_gate(gate_up_out)

    out_per_sample_g = _grouped_linear(
        gated_out,
        selected_down,
        selected_down_bias,
        offsets,
        is_transposed=self.is_transposed,
    )

    out_per_sample_g = out_per_sample_g * sample_weights_g.unsqueeze(-1)

    out_per_sample = out_per_sample_g[inv_perm]

    final_hidden_states = out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(
        dim=1
    )

    return final_hidden_states.to(hidden_states.dtype)


def patch_moe_histc():
    moe_module.grouped_mm_experts_forward = _patched_grouped_mm_experts_forward
    moe_module.ALL_EXPERTS_FUNCTIONS._global_mapping["grouped_mm"] = (
        _patched_grouped_mm_experts_forward
    )
