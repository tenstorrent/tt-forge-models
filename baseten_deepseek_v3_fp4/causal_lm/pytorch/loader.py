# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Baseten DeepSeek-V3-FP4 model loader implementation for causal language modeling.

This loads the FP4-quantized variant of DeepSeek-V3-0324 published by Baseten,
quantized with NVIDIA TensorRT Model Optimizer. Uses reduced MoE configuration
for testing since the full 397B parameter model requires multi-GPU hosts with
TensorRT-LLM to run.
"""

import types
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

# transformers 5.x removed DynamicCache.get_usable_length(); restore it as a
# compatibility shim so that the remote modeling_deepseek.py (which was written
# against an older transformers API) continues to work.
if not hasattr(DynamicCache, "get_usable_length"):

    def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        """Compatibility shim for transformers <5.x get_usable_length API."""
        previous_seq_length = self.get_seq_length(layer_idx)
        max_length = self.get_max_cache_shape()
        # get_max_cache_shape returns -1 for unbounded DynamicCache
        if max_length > 0 and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    DynamicCache.get_usable_length = _get_usable_length


def _moe_infer_batched(self, x, topk_ids, topk_weight):
    """Batched-matmul replacement for DeepseekV3MoE.moe_infer.

    The original moe_infer converts tokens_per_expert to numpy for loop
    control, which breaks torch.compile / XLA tracing.  This version stacks
    all expert weights into [E, *, *] tensors and uses a single bmm to
    evaluate every expert simultaneously, eliminating all data-dependent
    control flow.
    """
    T, H = x.shape
    E = self.experts_per_rank
    experts = self.experts[
        self.ep_rank * E : (self.ep_rank + 1) * E
    ]

    # Stack weights: [E, I, H] / [E, H, I]
    gate_w = torch.stack([e.gate_proj.weight for e in experts], dim=0)  # [E, I, H]
    up_w = torch.stack([e.up_proj.weight for e in experts], dim=0)      # [E, I, H]
    down_w = torch.stack([e.down_proj.weight for e in experts], dim=0)  # [E, H, I]

    x_fp = x.to(gate_w.dtype)
    x_exp = x_fp.unsqueeze(0).expand(E, -1, -1)          # [E, T, H]
    gate_out = torch.bmm(x_exp, gate_w.transpose(1, 2))  # [E, T, I]
    up_out = torch.bmm(x_exp, up_w.transpose(1, 2))      # [E, T, I]
    act = torch.nn.functional.silu(gate_out) * up_out    # [E, T, I]
    expert_outs = torch.bmm(act, down_w.transpose(1, 2)) # [E, T, H]

    # Build per-token routing weights: [T, E]
    routing = torch.zeros(T, E, dtype=topk_weight.dtype, device=x.device)
    routing.scatter_add_(1, topk_ids, topk_weight)

    # Weighted sum over experts: [T, E, H] * [T, E, 1] → [T, H]
    final_out = (
        expert_outs.permute(1, 0, 2) * routing.unsqueeze(-1).to(expert_outs.dtype)
    ).sum(dim=1)
    return final_out.to(x.dtype)


class ModelVariant(StrEnum):
    """Available Baseten DeepSeek-V3-FP4 model variants for causal language modeling."""

    BASETEN_DEEPSEEK_V3_FP4 = "V3_FP4"


class ModelLoader(ForgeModel):
    """Baseten DeepSeek-V3-FP4 model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.BASETEN_DEEPSEEK_V3_FP4: LLMModelConfig(
            pretrained_model_name="baseten/DeepSeek-V3-FP4",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASETEN_DEEPSEEK_V3_FP4

    sample_text = "What is machine learning?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Baseten-DeepSeek-V3-FP4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )

        # Reduce model dimensions for testing since the full 397B
        # MoE model is too large to load directly.
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        else:
            config.num_hidden_layers = 6
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

        for module in model.modules():
            if module.__class__.__name__ == "DeepseekV3MoE":
                module.moe_infer = types.MethodType(_moe_infer_batched, module)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
