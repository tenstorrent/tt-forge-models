# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TeleChat2 model loader implementation for causal language modeling.
"""
import os
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.dynamic_module_utils import get_imports
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


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


@contextmanager
def patch_cuda_and_flash_attn():
    """Temporarily disable TorchFunctionOverride, patch .cuda(), and mock flash_attn.

    The remote TeleChat2 modeling code has two issues on CPU-only systems:
    1. RotaryEmbedding.__init__ calls .cuda() which fails without CUDA.
       The active TorchFunctionOverride dispatches .cuda() through the C-level
       implementation, bypassing Python-level patches, so we must exit the mode.
    2. FlashSelfAttention.__init__ asserts flash_attn is installed.
       We mock the flash_attn module so the assertion passes, and set
       config.flash_attn=False to avoid actually using it in forward().
    """
    from tt_torch.torch_overrides import torch_function_override

    # Exit TorchFunctionOverride so .cuda() patch takes effect
    torch_function_override.__exit__(None, None, None)
    original_cuda = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *args, **kwargs: self

    # Mock flash_attn module to prevent assertion in FlashSelfAttention.__init__
    had_flash_attn = "flash_attn" in sys.modules
    had_flash_attn_interface = "flash_attn.flash_attn_interface" in sys.modules
    orig_flash_attn = sys.modules.get("flash_attn")
    orig_flash_attn_interface = sys.modules.get("flash_attn.flash_attn_interface")
    if not had_flash_attn:
        mock_interface = MagicMock()
        mock_interface.flash_attn_unpadded_func = MagicMock()
        mock_interface.flash_attn_varlen_func = MagicMock()
        sys.modules["flash_attn"] = MagicMock()
        sys.modules["flash_attn.flash_attn_interface"] = mock_interface

    # Clear cached transformers module so it re-imports with mock flash_attn
    stale_keys = [
        k for k in sys.modules if "telechat2" in k.lower() and "modeling" in k.lower()
    ]
    stale_modules = {k: sys.modules.pop(k) for k in stale_keys}
    try:
        yield
    finally:
        torch.Tensor.cuda = original_cuda
        torch_function_override.__enter__()
        sys.modules.update(stale_modules)
        if not had_flash_attn:
            del sys.modules["flash_attn"]
        else:
            sys.modules["flash_attn"] = orig_flash_attn
        if not had_flash_attn_interface:
            if "flash_attn.flash_attn_interface" in sys.modules:
                del sys.modules["flash_attn.flash_attn_interface"]
        else:
            sys.modules["flash_attn.flash_attn_interface"] = orig_flash_attn_interface


class ModelVariant(StrEnum):
    """Available TeleChat2 model variants for causal language modeling."""

    TELECHAT2_3B = "telechat2_3b"


class ModelLoader(ForgeModel):
    """TeleChat2 model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.TELECHAT2_3B: LLMModelConfig(
            pretrained_model_name="Tele-AI/TeleChat2-3B",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.TELECHAT2_3B

    # Shared configuration parameters
    sample_text = "What is the meaning of life?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="TeleChat2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            **tokenizer_kwargs,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        with (
            patch(
                "transformers.dynamic_module_utils.get_imports",
                fixed_get_imports,
            ),
            patch_cuda_and_flash_attn(),
        ):
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.flash_attn = False
            if self.num_layers is not None:
                config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

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
            [self.sample_text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
