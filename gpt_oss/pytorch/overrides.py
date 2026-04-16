# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F


def override_gpt_oss_modules(model):
    """
    De-interleave gate_up_proj into separate gate_proj and up_proj, and
    patch forward to use the batched BMM path for both training and inference.
    """
    import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss_mod

    for module in model.modules():
        if isinstance(module, gpt_oss_mod.GptOssExperts):
            _deinterleave_expert_weights(module)
        if isinstance(module, gpt_oss_mod.GptOssTopKRouter):
            module.forward = _expert_router_forward.__get__(module, type(module))
        if isinstance(module, gpt_oss_mod.GptOssMLP):
            module.forward = _mlp_forward.__get__(module, type(module))


def build_deinterleaved_shard_specs(model):
    """Build shard specs for the de-interleaved GPT-OSS model.

    Follows Megatron-style 1D tensor parallelism.
    Expert weights are sharded on the expert dimension only.

    Args:
        model: GPT-OSS model with de-interleaved expert weights.

    Returns:
        Dictionary mapping model parameters to their shard specifications.
    """
    shard_specs = {}

    # Embedding and output layer: replicatedI th
    shard_specs[model.model.embed_tokens.weight] = (None, None)
    shard_specs[model.model.norm.weight] = (None,)
    shard_specs[model.lm_head.weight] = (None, None)

    for layer in model.model.layers:
        # Attention: column-parallel for q/k/v, row-parallel for o
        shard_specs[layer.self_attn.q_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.q_proj.bias] = ("model",)
        shard_specs[layer.self_attn.k_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.k_proj.bias] = ("model",)
        shard_specs[layer.self_attn.v_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.v_proj.bias] = ("model",)
        shard_specs[layer.self_attn.o_proj.weight] = (None, "model")
        shard_specs[layer.self_attn.o_proj.bias] = (None,)

        # Sinks: replicated
        shard_specs[layer.self_attn.sinks] = (None,)

        # Router: replicated
        shard_specs[layer.mlp.router.weight] = (None, None)

        # De-interleaved expert weights: shard on expert dimension only
        shard_specs[layer.mlp.experts.gate_proj] = ("model", None, None)
        shard_specs[layer.mlp.experts.up_proj] = ("model", None, None)
        shard_specs[layer.mlp.experts.down_proj] = ("model", None, None)
        shard_specs[layer.mlp.experts.gate_proj_bias] = ("model", None)
        shard_specs[layer.mlp.experts.up_proj_bias] = ("model", None)
        shard_specs[layer.mlp.experts.down_proj_bias] = ("model", None)

        # Layer norms: replicated
        shard_specs[layer.input_layernorm.weight] = (None,)
        shard_specs[layer.post_attention_layernorm.weight] = (None,)

    return shard_specs


def _deinterleave_expert_weights(experts):
    with torch.no_grad():
        gate_proj_data = experts.gate_up_proj.data[:, :, ::2].contiguous()
        up_proj_data = experts.gate_up_proj.data[:, :, 1::2].contiguous()
        gate_bias_data = experts.gate_up_proj_bias.data[:, ::2].contiguous()
        up_bias_data = experts.gate_up_proj_bias.data[:, 1::2].contiguous()

    del experts.gate_up_proj
    del experts.gate_up_proj_bias

    experts.gate_proj = nn.Parameter(gate_proj_data)
    experts.up_proj = nn.Parameter(up_proj_data)
    experts.gate_proj_bias = nn.Parameter(gate_bias_data)
    experts.up_proj_bias = nn.Parameter(up_bias_data)

    experts.forward = _deinterleaved_experts_forward.__get__(experts, type(experts))


def _deinterleaved_experts_forward(
    self, hidden_states, router_indices=None, routing_weights=None
):
    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)
    num_experts = routing_weights.shape[1]

    hidden_states = hidden_states.repeat(num_experts, 1)
    hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)

    gate = torch.bmm(hidden_states, self.gate_proj) + self.gate_proj_bias[..., None, :]
    up = torch.bmm(hidden_states, self.up_proj) + self.up_proj_bias[..., None, :]

    gate = gate.clamp(min=None, max=self.limit)
    up = up.clamp(min=-self.limit, max=self.limit)
    glu = gate * torch.sigmoid(gate * self.alpha)
    next_states = torch.bmm(((up + 1) * glu), self.down_proj)
    next_states = next_states + self.down_proj_bias[..., None, :]
    next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
    next_states = (
        next_states
        * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
    )
    next_states = next_states.sum(dim=0)
    return next_states


def _expert_router_forward(self, hidden_states):
    flat = hidden_states.reshape(-1, self.hidden_dim).float()

    # Force float32 for router.
    router_logits = F.linear(flat, self.weight.float(), self.bias.float())
    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
    router_top_value = F.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)

    router_scores = torch.zeros_like(router_logits).scatter_(
        1, router_indices, router_top_value
    )

    return router_scores.to(hidden_states.dtype), router_indices


def _mlp_forward(self, hidden_states):
    router_scores, router_indices = self.router(hidden_states)
    routed_out = self.experts(
        hidden_states, router_indices=router_indices, routing_weights=router_scores
    )
    return routed_out, router_scores
