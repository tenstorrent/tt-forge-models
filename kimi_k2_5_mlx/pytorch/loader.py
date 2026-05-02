# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kimi K2.5 MLX quantized model loader implementation.

Loads the inferencerlabs/Kimi-K2.5-MLX quantized variants of the
Kimi K2.5 (DeepSeek V3 text backbone) for causal language modeling.
"""
import os
import sys
from typing import Optional
from unittest.mock import patch

import torch

# Patch missing functions before importing model code that depends on them.
# The model's remote code was written for an older transformers that included
# these helpers; newer versions removed them.
import transformers.utils
import transformers.utils.import_utils

if not hasattr(transformers.utils, "is_flash_attn_greater_or_equal_2_10"):

    def _is_flash_attn_gte_2_10():
        return False

    transformers.utils.is_flash_attn_greater_or_equal_2_10 = _is_flash_attn_gte_2_10
    sys.modules["transformers.utils"].__dict__[
        "is_flash_attn_greater_or_equal_2_10"
    ] = _is_flash_attn_gte_2_10

if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):

    def _is_torch_fx_available():
        return False

    transformers.utils.import_utils.is_torch_fx_available = _is_torch_fx_available
    sys.modules["transformers.utils.import_utils"].__dict__[
        "is_torch_fx_available"
    ] = _is_torch_fx_available

# Patch DynamicCache.from_legacy_cache removed in newer transformers
from transformers.cache_utils import DynamicCache

if not hasattr(DynamicCache, "from_legacy_cache"):

    @classmethod  # type: ignore[misc]
    def _from_legacy_cache(cls, past_key_values=None):
        cache = cls()
        if past_key_values is not None:
            for layer_idx, (key, value) in enumerate(past_key_values):
                cache.update(key, value, layer_idx)
        return cache

    DynamicCache.from_legacy_cache = _from_legacy_cache

if not hasattr(DynamicCache, "to_legacy_cache"):

    def _to_legacy_cache(self):
        legacy_cache = []
        for layer in self.layers:
            legacy_cache.append((layer.keys, layer.values))
        return legacy_cache

    DynamicCache.to_legacy_cache = _to_legacy_cache

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.dynamic_module_utils import get_imports

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


class ModelVariant(StrEnum):
    """Available Kimi K2.5 MLX model variants."""

    KIMI_K2_5_MLX_3_6BIT = "Kimi-K2.5-MLX-3.6bit"
    KIMI_K2_5_MLX_4_2BIT = "Kimi-K2.5-MLX-4.2bit"


class ModelLoader(ForgeModel):
    """Kimi K2.5 MLX quantized model loader implementation."""

    _VARIANTS = {
        ModelVariant.KIMI_K2_5_MLX_3_6BIT: None,
        ModelVariant.KIMI_K2_5_MLX_4_2BIT: None,
    }

    DEFAULT_VARIANT = ModelVariant.KIMI_K2_5_MLX_3_6BIT

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
    ):
        super().__init__(variant)
        if self._variant == ModelVariant.KIMI_K2_5_MLX_4_2BIT:
            self.model_name = "inferencerlabs/Kimi-K2.5-MLX-4.2bit"
        else:
            self.model_name = "inferencerlabs/Kimi-K2.5-MLX-3.6bit"
        self.tokenizer = None
        self.text = "What is machine learning?"

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Kimi-K2.5-MLX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Kimi K2.5 MLX quantized model (DeepSeek V3 architecture)."""
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            model_kwargs = {
                "attn_implementation": "eager",
                "trust_remote_code": True,
            }
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
            model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        return model

    def load_inputs(self, batch_size=1):
        if self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(self.text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
