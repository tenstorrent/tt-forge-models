# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SDAR (Synergy of Diffusion and AutoRegression) model loader implementation for causal language modeling.
"""
import os
from typing import Optional
from unittest.mock import patch

import sys

import torch
import transformers.cache_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.dynamic_module_utils import get_imports, get_relative_imports

# transformers 5.x removed SlidingWindowCache; inject a stub so the model's remote code loads.
if not hasattr(transformers.cache_utils, "SlidingWindowCache"):
    transformers.cache_utils.SlidingWindowCache = type(
        "SlidingWindowCache", (transformers.cache_utils.Cache,), {}
    )
    sys.modules[
        "transformers.cache_utils"
    ].SlidingWindowCache = transformers.cache_utils.SlidingWindowCache

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


_MISSING_MODULES = {"fused_linear_diffusion_cross_entropy"}


def _fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def _fixed_get_relative_imports(filename: str | os.PathLike) -> list[str]:
    return [m for m in get_relative_imports(filename) if m not in _MISSING_MODULES]


class ModelVariant(StrEnum):
    """Available SDAR model variants for causal language modeling."""

    SDAR_1_7B_CHAT = "1.7B_Chat"
    SDAR_1_7B_CHAT_B32 = "1.7B_Chat_b32"


class ModelLoader(ForgeModel):
    """SDAR model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.SDAR_1_7B_CHAT: LLMModelConfig(
            pretrained_model_name="JetLM/SDAR-1.7B-Chat",
            max_length=128,
        ),
        ModelVariant.SDAR_1_7B_CHAT_B32: LLMModelConfig(
            pretrained_model_name="JetLM/SDAR-1.7B-Chat-b32",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SDAR_1_7B_CHAT

    sample_text = "Explain what reinforcement learning is in simple terms."

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
            model="SDAR",
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

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        with patch(
            "transformers.dynamic_module_utils.get_imports", _fixed_get_imports
        ), patch(
            "transformers.dynamic_module_utils.get_relative_imports",
            _fixed_get_relative_imports,
        ):
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name,
                trust_remote_code=True,
                attn_implementation="eager",
                **model_kwargs,
            ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )

        return self.config
