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

from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import DynamicCache

# transformers 5.x removed get_usable_length; patch it back for compatibility
# with models that still use it in their custom modeling code
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = (
        lambda self, new_seq_len, layer_idx=0: self.get_seq_length(layer_idx)
    )

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
