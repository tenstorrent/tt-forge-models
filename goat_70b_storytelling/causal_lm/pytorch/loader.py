# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GOAT-AI/GOAT-70B-Storytelling model loader implementation for causal language modeling.
"""

import os
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
    """Available GOAT-70B-Storytelling model variants."""

    GOAT_70B_STORYTELLING = "70B_Storytelling"


class ModelLoader(ForgeModel):
    """GOAT-AI/GOAT-70B-Storytelling model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GOAT_70B_STORYTELLING: LLMModelConfig(
            pretrained_model_name="GOAT-AI/GOAT-70B-Storytelling",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GOAT_70B_STORYTELLING

    sample_text = "Once upon a time in a distant jungle, a lone explorer stumbled upon"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="GOAT-70B-Storytelling",
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
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        use_random_weights = os.environ.get("TT_RANDOM_WEIGHTS") or os.environ.get(
            "TT_COMPILE_ONLY_SYSTEM_DESC"
        )

        if self.num_layers is not None or use_random_weights:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            if self.num_layers is not None:
                config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        if use_random_weights:
            config = model_kwargs.pop("config")
            torch_dtype = model_kwargs.pop("torch_dtype", None)
            model = AutoModelForCausalLM.from_config(config)
            if torch_dtype is not None:
                model = model.to(torch_dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
        model.eval()

        self.config = model.config
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
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
