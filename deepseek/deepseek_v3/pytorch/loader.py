# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V3 model loader implementation for causal language modeling.
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = (
        lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
    )


def _patch_moe_layers(model) -> None:
    """Replace moe_infer with a batched expert matmul (no numpy / no device transfer).

    Stacks all expert gate/up/down weights into 3D tensors and runs them as a
    single bmm each, then applies routing weights via scatter_add + einsum.
    This produces O(1) graph nodes regardless of expert count, avoiding the
    O(256) sequential graph that causes multi-minute MLIR compilation.
    """

    def _batched_moe_infer(self_moe, x, topk_ids, topk_weight):
        num_tokens, hidden = x.shape
        num_experts = self_moe.experts_per_rank

        # Stack expert weights: each [inter, hidden] → [E, inter, hidden]
        gate_w = torch.stack([e.gate_proj.weight for e in self_moe.experts])
        up_w = torch.stack([e.up_proj.weight for e in self_moe.experts])
        down_w = torch.stack([e.down_proj.weight for e in self_moe.experts])

        # x: [T, hidden] → [E, T, hidden] for batched matmul
        x_exp = x.unsqueeze(0).expand(num_experts, -1, -1).contiguous()

        # [E, T, inter]
        gate_out = torch.bmm(x_exp, gate_w.transpose(1, 2))
        up_out = torch.bmm(x_exp, up_w.transpose(1, 2))
        hidden_states = self_moe.experts[0].act_fn(gate_out) * up_out
        # [E, T, hidden]
        expert_out = torch.bmm(hidden_states, down_w.transpose(1, 2))

        # Routing: scatter topk weights into [T, E]
        routing = torch.zeros(
            num_tokens, num_experts, dtype=topk_weight.dtype, device=x.device
        ).scatter_add(1, topk_ids, topk_weight.to(topk_weight.dtype))

        # Weighted sum over experts: [T, E] x [E, T, H] -> [T, H]
        # Cast back to x.dtype: routing is float32 (from router scores) and
        # would otherwise upcast expert_out (bf16) → float32, breaking lm_head.
        # expert_out.permute(1, 0, 2): [T, E, H]
        return (routing.unsqueeze(-1) * expert_out.permute(1, 0, 2)).sum(dim=1).to(x.dtype)

    for module in model.modules():
        if hasattr(module, "moe_infer") and hasattr(module, "experts_per_rank"):
            # Verify expert structure is compatible before patching
            experts = list(module.experts)
            if experts and hasattr(experts[0], "gate_proj") and hasattr(experts[0], "up_proj"):
                type(module).moe_infer = _batched_moe_infer


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


class ModelVariant(StrEnum):
    """Available DeepSeek-V3 model variants."""

    TINY_RANDOM = "Tiny_Random"


class ModelLoader(ForgeModel):
    """DeepSeek-V3 model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM: LLMModelConfig(
            pretrained_model_name="yujiepan/deepseek-v3-tiny-random",
            max_length=2048,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM

    sample_text = "What is machine learning?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DeepSeek-V3",
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
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        _patch_moe_layers(model)

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(self.sample_text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
