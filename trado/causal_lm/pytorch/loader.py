# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TraDo Causal LM model loader implementation
"""

import sys
import types
import torch
from unittest.mock import patch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.dynamic_module_utils import get_imports
from typing import Optional


def _fixed_get_imports(filename) -> list[str]:
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def _rms_norm_fn(
    x,
    weight,
    bias=None,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    eps=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    residual_in_fp32=False,
    is_rms_norm=True,
    num_groups=1,
    norm_before_gate=True,
    gate=None,
):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    out = weight * x.to(weight.dtype)
    if bias is not None:
        out = out + bias
    if prenorm:
        return out, residual if residual is not None else x
    return out


def _inject_flash_attn_mock():
    if "flash_attn" not in sys.modules:
        flash_attn = types.ModuleType("flash_attn")
        ops = types.ModuleType("flash_attn.ops")
        triton = types.ModuleType("flash_attn.ops.triton")
        layer_norm = types.ModuleType("flash_attn.ops.triton.layer_norm")
        layer_norm.rms_norm_fn = _rms_norm_fn
        flash_attn.ops = ops
        ops.triton = triton
        triton.layer_norm = layer_norm
        sys.modules["flash_attn"] = flash_attn
        sys.modules["flash_attn.ops"] = ops
        sys.modules["flash_attn.ops.triton"] = triton
        sys.modules["flash_attn.ops.triton.layer_norm"] = layer_norm


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
    """Available TraDo model variants for causal language modeling."""

    TRADO_4B_INSTRUCT = "4b_instruct"
    TRADO_8B_THINKING = "8b_thinking"


class ModelLoader(ForgeModel):
    """TraDo model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.TRADO_4B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Gen-Verse/TraDo-4B-Instruct",
            max_length=256,
        ),
        ModelVariant.TRADO_8B_THINKING: LLMModelConfig(
            pretrained_model_name="Gen-Verse/TraDo-8B-Thinking",
            max_length=256,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.TRADO_4B_INSTRUCT

    # Shared configuration parameters
    sample_text = "What is the sum of 2 and 3?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="trado",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
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
        """Load and return the TraDo model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The TraDo model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

        _inject_flash_attn_mock()
        with patch("transformers.dynamic_module_utils.get_imports", _fixed_get_imports):
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
        model.eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the TraDo model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        """Load and return the configuration for the TraDo model variant.

        Returns:
            The configuration object for the TraDo model.
        """
        _inject_flash_attn_mock()
        with patch("transformers.dynamic_module_utils.get_imports", _fixed_get_imports):
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
            )

        return self.config
