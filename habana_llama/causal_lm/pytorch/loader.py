# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Habana/llama model loader implementation for causal language modeling.
"""

from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """Available Habana/llama model variants for causal language modeling."""

    HABANA_LLAMA = "Habana_llama"


class ModelLoader(ForgeModel):
    """Habana/llama model loader implementation for causal language modeling tasks.

    Habana/llama is a Hugging Face repository published by Habana that contains
    only a GaudiConfig for running Llama models on Habana Gaudi processors
    (HPU). It ships no model weights of its own, so this loader instantiates
    the upstream huggyllama/llama-7b checkpoint referenced by Habana's example
    scripts.
    """

    _VARIANTS = {
        ModelVariant.HABANA_LLAMA: LLMModelConfig(
            pretrained_model_name="huggyllama/llama-7b",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HABANA_LLAMA

    sample_text = "Hello there fellow traveler"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Habana/llama",
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
            **tokenizer_kwargs,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
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
