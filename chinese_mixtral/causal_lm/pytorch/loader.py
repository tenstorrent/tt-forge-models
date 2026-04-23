# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chinese Mixtral model loader implementation for causal language modeling.
"""

import os
import shutil
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

_MODEL_SPACE_THRESHOLD = 30 * 1024 * 1024 * 1024  # 30 GB
_FALLBACK_CACHE_DIR = "/tmp/hf_cache_chinese_mixtral"


def _get_cache_dir():
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = os.path.join(hf_home, "hub")
    try:
        os.makedirs(hub_dir, exist_ok=True)
        if shutil.disk_usage(hub_dir).free >= _MODEL_SPACE_THRESHOLD:
            return None
    except OSError:
        pass
    os.makedirs(_FALLBACK_CACHE_DIR, exist_ok=True)
    return _FALLBACK_CACHE_DIR


class ModelVariant(StrEnum):
    """Available Chinese Mixtral model variants."""

    CHINESE_MIXTRAL_INSTRUCT = "Chinese_Mixtral_Instruct"


class ModelLoader(ForgeModel):
    """Chinese Mixtral model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.CHINESE_MIXTRAL_INSTRUCT: ModelConfig(
            pretrained_model_name="hfl/chinese-mixtral-instruct",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHINESE_MIXTRAL_INSTRUCT

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Chinese Mixtral",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        cache_dir = _get_cache_dir()
        if cache_dir is not None:
            tokenizer_kwargs["cache_dir"] = cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        cache_dir = _get_cache_dir()
        if cache_dir is not None:
            model_kwargs["cache_dir"] = cache_dir
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name,
                **({"cache_dir": cache_dir} if cache_dir is not None else {}),
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        test_input = "What is the capital of France?"

        inputs = self.tokenizer(test_input, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])

    def load_config(self):
        cache_dir = _get_cache_dir()
        config_kwargs = {}
        if cache_dir is not None:
            config_kwargs["cache_dir"] = cache_dir
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, **config_kwargs
        )
        return self.config
