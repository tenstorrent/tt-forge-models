# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.7-Flash-Grande-Heretic-UNCENSORED-42B-A3B GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata as _importlib_metadata
from typing import Optional

import torch
import transformers.modeling_gguf_pytorch_utils as _gguf_pytorch_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def _is_gguf_available(min_version: str = "0.10.0") -> bool:
    # transformers caches PACKAGE_DISTRIBUTION_MAPPING at import time, so
    # dynamically-installed gguf is not found. Use importlib.metadata directly.
    try:
        from packaging.version import Version

        return Version(_importlib_metadata.version("gguf")) >= Version(min_version)
    except Exception:
        return False


_gguf_pytorch_utils.is_gguf_available = _is_gguf_available


def _patch_transformers_deepseek_v2_gguf():
    """Monkey-patch transformers to add deepseek_v2 GGUF tokenizer support."""
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )

    if "deepseek_v2" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["deepseek_v2"] = GGUFQwen2Converter


_patch_transformers_deepseek_v2_gguf()

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
    """Available GLM-4.7-Flash-Grande-Heretic-UNCENSORED-42B-A3B GGUF model variants for causal language modeling."""

    GLM_4_7_FLASH_GRANDE_HERETIC_UNCENSORED_42B_A3B_GGUF = (
        "4.7_Flash_Grande_Heretic_UNCENSORED_42B_A3B_GGUF"
    )


class ModelLoader(ForgeModel):
    """GLM-4.7-Flash-Grande-Heretic-UNCENSORED-42B-A3B GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GLM_4_7_FLASH_GRANDE_HERETIC_UNCENSORED_42B_A3B_GGUF: LLMModelConfig(
            pretrained_model_name="DavidAU/GLM-4.7-Flash-Grande-Heretic-UNCENSORED-42B-A3B-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4_7_FLASH_GRANDE_HERETIC_UNCENSORED_42B_A3B_GGUF

    GGUF_FILE = "GLM-4.7-30B-A3B-20-2-Heretic-30B-A3B-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language models."

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
            model="GLM-4.7-Flash-Grande-Heretic-UNCENSORED-42B-A3B GGUF",
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
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

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
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
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
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
