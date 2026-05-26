# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama GGUF model loader.

Loads bartowski/Llama-3.2-3B-Instruct-GGUF quantized variants via the
transformers ``gguf_file`` integration. Weights are dequantized into a
standard ``LlamaForCausalLM`` so downstream compilation behaves like any
other Hugging Face PyTorch causal LM.
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


# Bartowski's GGUF metadata for Llama 3.2 reports rope_type="default", which
# omits the Llama-3 frequency scaling (factor=32, llama3 type). Without it the
# model runs with a "default" rope it was never trained on, producing noisy
# outputs that hurt compile-time PCC. These are the canonical Llama 3.x rope
# parameters from the original meta-llama config.
_LLAMA3_ROPE_PARAMETERS = {
    "rope_type": "llama3",
    "rope_theta": 500000.0,
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
}

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
from ....tools.utils import cast_input_to_type, pad_inputs


class ModelVariant(StrEnum):
    """GGUF variants published by bartowski for Llama-3.2-3B-Instruct.

    F16 is the unquantized GGUF (original weights in fp16). It exercises the
    transformers GGUF loading path while preserving safetensors-equivalent
    numerics, which keeps post-compile PCC at the same level as the
    safetensors loader.
    """

    LLAMA_3_2_3B_INSTRUCT_F16 = "3.2_3B_Instruct_F16"


# Map variant to GGUF filename within the repo.
_GGUF_FILES = {
    ModelVariant.LLAMA_3_2_3B_INSTRUCT_F16: "Llama-3.2-3B-Instruct-f16.gguf",
}


class ModelLoader(ForgeModel):
    """Loader for bartowski/Llama-3.2-3B-Instruct-GGUF dequantized via transformers."""

    _VARIANTS = {
        ModelVariant.LLAMA_3_2_3B_INSTRUCT_F16: LLMModelConfig(
            pretrained_model_name="bartowski/Llama-3.2-3B-Instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_2_3B_INSTRUCT_F16

    sample_text = "Hey how are you doing today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LlamaGGUF",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _gguf_file(self) -> str:
        return _GGUF_FILES[self._variant]

    def _load_tokenizer(self):
        pretrained_model_name = self._variant_config.pretrained_model_name
        # Tokenizer is reconstructed from the GGUF metadata.
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, gguf_file=self._gguf_file()
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Llama GGUF model dequantized into a torch module."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"gguf_file": self._gguf_file()}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # Restore Llama-3 rope (lost in GGUF metadata) and rebuild the rotary
        # embedding so generation matches the safetensors model.
        model.config.rope_parameters = dict(_LLAMA3_ROPE_PARAMETERS)
        if getattr(model.config, "rope_scaling", None) is not None:
            model.config.rope_scaling = dict(_LLAMA3_ROPE_PARAMETERS)
        device = next(model.parameters()).device
        model.model.rotary_emb = LlamaRotaryEmbedding(model.config).to(device)

        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return tokenized + padded sample inputs suitable for causal LM."""
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(self.sample_text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs
