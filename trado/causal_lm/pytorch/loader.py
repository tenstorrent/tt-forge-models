# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TraDo Causal LM model loader implementation
"""

import sys
import types
import importlib
import importlib.machinery

import torch
from typing import Optional


def _ensure_flash_attn_available():
    try:
        importlib.import_module("flash_attn")
        return
    except (ImportError, ModuleNotFoundError):
        pass

    def _rms_norm_fn(
        x,
        weight,
        bias=None,
        residual=None,
        prenorm=False,
        eps=1e-6,
        residual_in_fp32=False,
        is_rms_norm=True,
    ):
        dtype = x.dtype
        x = x.float()
        if residual is not None:
            residual = residual.float()
            x = x + residual
        residual_out = x if prenorm else None
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        x = x.to(dtype) * weight
        if bias is not None:
            x = x + bias
        if prenorm:
            return x, residual_out
        return x

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__version__ = "0.0.0"
    flash_attn.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    flash_attn.flash_attn_func = None
    flash_attn.flash_attn_varlen_func = None

    bert_padding = types.ModuleType("flash_attn.bert_padding")
    bert_padding.index_first_axis = None
    bert_padding.pad_input = None
    bert_padding.unpad_input = None
    flash_attn.bert_padding = bert_padding

    ops = types.ModuleType("flash_attn.ops")
    triton_mod = types.ModuleType("flash_attn.ops.triton")
    layer_norm = types.ModuleType("flash_attn.ops.triton.layer_norm")
    layer_norm.rms_norm_fn = _rms_norm_fn
    triton_mod.layer_norm = layer_norm
    ops.triton = triton_mod
    flash_attn.ops = ops

    sys.modules["flash_attn"] = flash_attn
    sys.modules["flash_attn.bert_padding"] = bert_padding
    sys.modules["flash_attn.ops"] = ops
    sys.modules["flash_attn.ops.triton"] = triton_mod
    sys.modules["flash_attn.ops.triton.layer_norm"] = layer_norm


_ensure_flash_attn_available()


def _ensure_sliding_window_cache():
    import transformers.cache_utils as cu

    if hasattr(cu, "SlidingWindowCache"):
        return

    class SlidingWindowCache(cu.StaticCache):
        def __init__(self, config, max_batch_size, max_cache_len=None, **kwargs):
            super().__init__(
                config, max_batch_size, max_cache_len=max_cache_len, **kwargs
            )

    cu.SlidingWindowCache = SlidingWindowCache


_ensure_sliding_window_cache()


def _ensure_loss_kwargs():
    import transformers.utils as tu

    if hasattr(tu, "LossKwargs"):
        return

    from typing import TypedDict

    class LossKwargs(TypedDict, total=False):
        num_items_in_batch: int

    tu.LossKwargs = LossKwargs


_ensure_loss_kwargs()


def _ensure_default_rope():
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope_parameters(
        config=None, device=None, seq_len=None, **kwargs
    ):
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        base = config.rope_theta
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


_ensure_default_rope()

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

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


class ModelLoader(ForgeModel):
    """TraDo model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.TRADO_4B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Gen-Verse/TraDo-4B-Instruct",
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

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = config.eos_token_id

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
            "config": config,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

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
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )

        return self.config
