# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AI21 Jamba Large 1.6 model loader implementation for causal language modeling.
"""

import os

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    JambaConfig,
    JambaForCausalLM,
)
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available AI21 Jamba Large 1.6 model variants for causal language modeling."""

    AI21_JAMBA_LARGE_1_6 = "Large_1.6"


class ModelLoader(ForgeModel):
    """AI21 Jamba Large 1.6 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.AI21_JAMBA_LARGE_1_6: LLMModelConfig(
            pretrained_model_name="ai21labs/AI21-Jamba-Large-1.6",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AI21_JAMBA_LARGE_1_6

    sample_text = "My name is Thomas and my main"

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
            model="AI21 Jamba Large 1.6",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        if os.environ.get("TT_RANDOM_WEIGHTS"):
            self.tokenizer = None
            return None

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def _get_config(self):
        if os.environ.get("TT_RANDOM_WEIGHTS"):
            config = JambaConfig()
        else:
            config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        return config

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = self._get_config()
        config.use_mamba_kernels = False

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            model = JambaForCausalLM(config)
            if dtype_override is not None:
                model = model.to(dtype_override)
        else:
            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs["use_mamba_kernels"] = False
            model_kwargs |= kwargs
            if self.num_layers is not None:
                model_kwargs["config"] = config
            model = AutoModelForCausalLM.from_pretrained(
                self._variant_config.pretrained_model_name, **model_kwargs
            )

        model = model.eval()
        self.config = model.config
        self.model = model
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

        inputs = self.tokenizer(
            [self.sample_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = self._get_config()
        return self.config
