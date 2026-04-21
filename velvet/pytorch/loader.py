# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Velvet model loader implementation for causal language modeling.
"""

import os

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    MistralConfig,
    MistralForCausalLM,
)
from typing import Optional
from ...tools.utils import generate_no_cache, pad_inputs
from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

_VELVET_2B_CONFIG = {
    "vocab_size": 126976,
    "max_position_embeddings": 32768,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 28,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-05,
    "head_dim": 64,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "rope_theta": 100000.0,
    "attention_dropout": 0.0,
}


class ModelVariant(StrEnum):
    """Available Velvet model variants."""

    VELVET_2B = "2B"


class ModelLoader(ForgeModel):
    """Velvet model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.VELVET_2B: LLMModelConfig(
            pretrained_model_name="Almawave/Velvet-2B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VELVET_2B

    sample_text = "Explain quantum computing in simple terms."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Velvet",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_config(self):
        if os.environ.get("TT_RANDOM_WEIGHTS"):
            return MistralConfig(**_VELVET_2B_CONFIG)
        return AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )

    def _load_tokenizer(self, dtype_override=None):
        if os.environ.get("TT_RANDOM_WEIGHTS"):
            self.tokenizer = None
            return None

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
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = self._get_config()

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            model = MistralForCausalLM(config)
            if dtype_override is not None:
                model = model.to(dtype_override)
        else:
            model_kwargs = {"trust_remote_code": True}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs
            model_kwargs["config"] = config
            model = MistralForCausalLM.from_pretrained(
                self._variant_config.pretrained_model_name, **model_kwargs
            )

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        max_length = self._variant_config.max_length

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            vocab_size = self._get_config().vocab_size
            input_ids = torch.randint(0, vocab_size, (batch_size, max_length))
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        messages = [{"role": "user", "content": self.sample_text}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        padded_inputs, seq_len = pad_inputs(inputs)
        self.seq_len = seq_len

        return padded_inputs

    def decode_output(self, max_new_tokens, model, inputs, tokenizer):
        generated_text = generate_no_cache(
            max_new_tokens, model, inputs, self.seq_len, tokenizer
        )
        return generated_text
