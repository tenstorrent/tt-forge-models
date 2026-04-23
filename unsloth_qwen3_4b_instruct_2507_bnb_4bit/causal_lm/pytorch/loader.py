# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
unsloth/Qwen3-4B-Instruct-2507-bnb-4bit model loader implementation for causal language modeling.
"""

import os
import shutil
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

_MIN_FREE_BYTES = 5 * 1024**3


def _get_cache_dir() -> str:
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_cache = os.path.join(hf_home, "hub")
    try:
        free = shutil.disk_usage(os.path.dirname(hub_cache)).free
    except Exception:
        free = 0
    return hub_cache if free >= _MIN_FREE_BYTES else "/tmp/hf_hub_cache"


class ModelVariant(StrEnum):
    """Available unsloth/Qwen3-4B-Instruct-2507-bnb-4bit model variants for causal LM."""

    QWEN3_4B_INSTRUCT_2507_BNB_4BIT = "Qwen3_4B_Instruct_2507_bnb_4bit"


class ModelLoader(ForgeModel):
    """unsloth/Qwen3-4B-Instruct-2507-bnb-4bit model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.QWEN3_4B_INSTRUCT_2507_BNB_4BIT: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen3-4B-Instruct-2507-bnb-4bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_4B_INSTRUCT_2507_BNB_4BIT

    sample_text = "Give me a short introduction to large language model."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="unsloth-Qwen3-4B-Instruct-2507-bnb-4bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"cache_dir": _get_cache_dir()}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"cache_dir": _get_cache_dir()}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._variant_config.max_length,
        )

        for key in inputs:
            if hasattr(inputs[key], "repeat_interleave"):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
