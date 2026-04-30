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
    """Replace moe_infer with a static per-expert masked matmul (no numpy / no device transfer)."""

    def _static_moe_infer(self_moe, x, topk_ids, topk_weight):
        out = torch.zeros_like(x)
        for i in range(self_moe.experts_per_rank):
            mask = topk_ids == i
            weight = (mask * topk_weight.to(x.dtype)).sum(dim=-1, keepdim=True)
            expert_out = self_moe.experts[i](x)
            out = out + expert_out * weight
        return out

    for module in model.modules():
        if hasattr(module, "moe_infer") and hasattr(module, "experts_per_rank"):
            type(module).moe_infer = _static_moe_infer


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
