# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM4-9B-Neon-v2 GGUF model loader implementation for causal language modeling.
"""

import importlib.metadata

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

from ....base import ForgeModel


def _patch_is_gguf_available():
    """Patch transformers' is_gguf_available to handle gguf packages lacking __version__.

    The gguf package does not define __version__. When gguf is installed after
    transformers is imported, PACKAGE_DISTRIBUTION_MAPPING is already cached without
    it, causing the version fallback to return 'N/A' and crash packaging.version.parse.
    We override is_gguf_available to use importlib.metadata.version directly.
    """
    try:
        import transformers.utils.import_utils as _tf_utils
        import transformers.modeling_gguf_pytorch_utils as _gguf_utils
        from packaging import version as _packaging_version

        def _patched_is_gguf_available(min_version=None):
            try:
                if min_version is None:
                    return _orig_is_gguf_available()
                return _orig_is_gguf_available(min_version)
            except Exception:
                try:
                    _min = (
                        min_version
                        if min_version is not None
                        else _tf_utils.GGUF_MIN_VERSION
                    )
                    gguf_ver = importlib.metadata.version("gguf")
                    return _packaging_version.parse(
                        gguf_ver
                    ) >= _packaging_version.parse(_min)
                except Exception:
                    return False

        _orig_is_gguf_available = _tf_utils.is_gguf_available
        _tf_utils.is_gguf_available = _patched_is_gguf_available
        _gguf_utils.is_gguf_available = _patched_is_gguf_available
    except Exception:
        pass


_patch_is_gguf_available()
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
    """Available GLM4-9B-Neon-v2 GGUF model variants for causal language modeling."""

    GLM4_9B_NEON_V2_Q4_K_M = "9B_Neon_v2_Q4_K_M"


class ModelLoader(ForgeModel):
    """GLM4-9B-Neon-v2 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GLM4_9B_NEON_V2_Q4_K_M: LLMModelConfig(
            pretrained_model_name="mradermacher/GLM4-9B-Neon-v2-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM4_9B_NEON_V2_Q4_K_M

    GGUF_FILE = "GLM4-9B-Neon-v2.Q4_K_M.gguf"

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
            model="GLM4-9B-Neon-v2 GGUF",
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
