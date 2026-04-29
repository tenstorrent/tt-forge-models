# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V2 model loader implementation for causal language modeling.

Uses reduced MoE configuration for testing since the full 236B parameter
model is too large to load directly.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

# transformers 5.x removed DynamicCache.get_usable_length; the custom
# modeling_deepseek.py still calls it, so restore as an alias.
if not hasattr(DynamicCache, "get_usable_length"):

    def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    DynamicCache.get_usable_length = _get_usable_length

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
)


class ModelLoader(ForgeModel):
    """DeepSeek-V2 model loader for causal language modeling."""

    def __init__(self, variant=None, num_layers: Optional[int] = None):
        super().__init__(variant)
        self.model_name = "deepseek-ai/DeepSeek-V2"
        self.tokenizer = None
        self.text = "The capital of France is"
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="DeepSeek-V2",
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
            config.num_hidden_layers = 2
        config.num_attention_heads = 16
        config.hidden_size = 1024
        config.num_key_value_heads = 16
        config.intermediate_size = 1024 * 4
        config.num_experts_per_tok = 2
        config.q_lora_rank = 256

        model_kwargs = {
            "attn_implementation": "eager",
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_config(config, **model_kwargs)
        model.eval()

        # Patch moe_infer to avoid .cpu().numpy() dispatch which fails under TT XLA.
        # Replace with static per-expert masked matmul so dynamo can unroll the loop.
        def _static_moe_infer(self_moe, x, topk_ids, topk_weight):
            out = torch.zeros_like(x)
            for i in range(self_moe.experts_per_rank):
                mask = (topk_ids == i)  # [num_tokens, num_experts_per_tok]
                weight = (mask * topk_weight.to(x.dtype)).sum(dim=-1, keepdim=True)
                expert_out = self_moe.experts[i](x)
                out = out + expert_out * weight
            return out

        for module in model.modules():
            if hasattr(module, "moe_infer") and hasattr(module, "experts_per_rank"):
                type(module).moe_infer = _static_moe_infer
                break  # patch the class once; all instances share it

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
