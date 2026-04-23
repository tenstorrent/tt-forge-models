# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EXAONE 3.5 GGUF model loader implementation for causal language modeling.
"""
import importlib.metadata
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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


def _fix_gguf_in_transformers():
    """Refresh stale transformers package metadata after dynamic gguf install.

    transformers caches importlib.metadata.packages_distributions() at module
    import time and wraps is_gguf_available() with @lru_cache.  When gguf is
    installed after transformers is imported (as the per-test RequirementsManager
    does), both caches are stale, causing version lookup to fall back to
    gguf.__version__ which the gguf package does not define, producing 'N/A'
    and an InvalidVersion error.
    """
    import transformers.utils.import_utils as _import_utils

    try:
        fresh = importlib.metadata.packages_distributions()
        if "gguf" in fresh:
            _import_utils.PACKAGE_DISTRIBUTION_MAPPING["gguf"] = fresh["gguf"]
    except Exception:
        pass
    try:
        _import_utils.is_gguf_available.cache_clear()
    except Exception:
        pass


class ModelVariant(StrEnum):
    """Available EXAONE 3.5 GGUF model variants for causal language modeling."""

    EXAONE_3_5_7_8B_INSTRUCT_GGUF = "3.5_7.8B_Instruct_GGUF"
    LGAI_EXAONE_3_5_7_8B_INSTRUCT_GGUF = "LGAI_3.5_7.8B_Instruct_GGUF"


class ModelLoader(ForgeModel):
    """EXAONE 3.5 GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.EXAONE_3_5_7_8B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="lmstudio-community/EXAONE-3.5-7.8B-Instruct-GGUF",
            max_length=128,
        ),
        ModelVariant.LGAI_EXAONE_3_5_7_8B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EXAONE_3_5_7_8B_INSTRUCT_GGUF

    GGUF_FILE = "EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf"

    # Download large GGUF files to /tmp to avoid filling up the main cache disk.
    GGUF_CACHE_DIR = "/tmp/hf_gguf_cache"

    sample_text = "Explain the basics of large language models."

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
            model="EXAONE 3.5 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        _fix_gguf_in_transformers()
        tokenizer_kwargs = {
            "gguf_file": self.GGUF_FILE,
            "cache_dir": self.GGUF_CACHE_DIR,
        }
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

        model_kwargs = {
            "gguf_file": self.GGUF_FILE,
            "cache_dir": self.GGUF_CACHE_DIR,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name,
                gguf_file=self.GGUF_FILE,
                cache_dir=self.GGUF_CACHE_DIR,
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
        _fix_gguf_in_transformers()
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self.GGUF_FILE,
            cache_dir=self.GGUF_CACHE_DIR,
        )
        return self.config
