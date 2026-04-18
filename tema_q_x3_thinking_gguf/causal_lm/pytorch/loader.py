# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tema Q-X3 Thinking GGUF model loader implementation for causal language modeling.
"""
import importlib
import importlib.metadata
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers.utils.import_utils as _import_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from packaging.version import Version
from typing import Optional


def _is_gguf_available(min_version: str = _import_utils.GGUF_MIN_VERSION) -> bool:
    try:
        sys.modules.pop("gguf", None)
        for key in list(sys.path_importer_cache):
            sys.path_importer_cache.pop(key, None)
        importlib.invalidate_caches()
        gguf = importlib.import_module("gguf")
        ver = getattr(gguf, "__version__", None)
        if ver is None:
            ver = importlib.metadata.version("gguf")
        return Version(ver) >= Version(min_version)
    except Exception:
        return False


def _patch_all_gguf_refs():
    _import_utils.is_gguf_available = _is_gguf_available
    _gguf_utils.is_gguf_available = _is_gguf_available
    for mod_name in list(sys.modules):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        if getattr(mod, "is_gguf_available", None) is not _is_gguf_available:
            if hasattr(mod, "is_gguf_available") and callable(
                getattr(mod, "is_gguf_available")
            ):
                try:
                    mod.is_gguf_available = _is_gguf_available
                except (AttributeError, TypeError):
                    pass


_patch_all_gguf_refs()

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
    """Available Tema Q-X3 Thinking GGUF model variants for causal language modeling."""

    TEMA_Q_X3_THINKING_I1_GGUF = "Tema_Q_X3_Thinking_i1_GGUF"


class ModelLoader(ForgeModel):
    """Tema Q-X3 Thinking GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.TEMA_Q_X3_THINKING_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Tema_Q-X3-Thinking-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TEMA_Q_X3_THINKING_I1_GGUF

    GGUF_FILE = "Tema_Q-X3-Thinking.i1-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language model."

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
            model="Tema Q-X3 Thinking GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _patch_all_gguf_refs()
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
        _patch_all_gguf_refs()
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
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
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
        _patch_all_gguf_refs()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
