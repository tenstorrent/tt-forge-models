# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.7 AWQ model loader implementation for causal language modeling.

Uses reduced MoE configuration for testing since the full ~181 GiB
AWQ-quantized model is too large to load directly.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
)


def _tt_static_glm4_moe_forward(experts_module, hidden_states, top_k_index, top_k_weights):
    # Per-expert masked matmul. Avoids 3D dynamic gather (L1 CB overflow from
    # batched_mm) and torch.histc on Int (unsupported in grouped_mm).
    # Loop over Python ints so dynamo unrolls into static F.linear calls.
    dtype = hidden_states.dtype
    out = torch.zeros_like(hidden_states)
    for expert_idx in range(experts_module.num_experts):
        mask = (top_k_index == expert_idx)  # [tokens, top_k]
        # Cast weight to model dtype — router may return float32 weights due to
        # its float32 e_score_correction_bias, which would promote out to float32.
        weight = (mask.to(dtype) * top_k_weights.to(dtype)).sum(dim=-1, keepdim=True)
        gate_up = F.linear(hidden_states, experts_module.gate_up_proj[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_expert = F.linear(
            experts_module.act_fn(gate) * up, experts_module.down_proj[expert_idx]
        )
        out = out + hidden_expert * weight
    return out.to(dtype)


ALL_EXPERTS_FUNCTIONS["tt_static_glm4_moe"] = _tt_static_glm4_moe_forward


class ModelLoader(ForgeModel):
    """GLM-4.7 AWQ model loader for causal language modeling."""

    def __init__(self, variant=None, num_layers: Optional[int] = None):
        super().__init__(variant)
        self.model_name = "QuantTrio/GLM-4.7-AWQ"
        self.tokenizer = None
        self.text = "What is machine learning?"
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="GLM-4.7-AWQ",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)

        # Reduce model dimensions for testing
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        else:
            config.num_hidden_layers = 6
        config.num_attention_heads = 16
        config.hidden_size = 1024
        config.num_key_value_heads = 8
        config.intermediate_size = 1024 * 4
        config.moe_intermediate_size = 1024
        config.num_experts_per_tok = 2
        config.n_routed_experts = 8

        model_kwargs = {
            "attn_implementation": "eager",
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        config._experts_implementation = "tt_static_glm4_moe"

        model = AutoModelForCausalLM.from_config(config, **model_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        return model

    def load_inputs(self, batch_size=1):
        if self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(self.text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
