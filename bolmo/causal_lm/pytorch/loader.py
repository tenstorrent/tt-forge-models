# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bolmo Causal LM model loader implementation.

Bolmo is a byte-level autoregressive language model developed by Ai2,
byteified from OLMo-2-0425-1B. The upstream repo is published with custom
modeling code so loading requires `trust_remote_code=True`.
"""

import transformers.utils.generic as _tug
import transformers.modeling_rope_utils as _rope_utils
from transformers.modeling_utils import PreTrainedModel as _PreTrainedModel
from xlstm.xlstm_large.model import mLSTMBackendConfig as _mLSTMBackendConfig

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional

_orig_check_model_inputs = _tug.check_model_inputs


def _check_model_inputs_compat(func=None):
    # modeling_bolmo.py calls @check_model_inputs() (factory style); patch to
    # support both @check_model_inputs and @check_model_inputs().
    if func is None:
        return _orig_check_model_inputs
    return _orig_check_model_inputs(func)


_tug.check_model_inputs = _check_model_inputs_compat


def _default_rope_init(config, device=None, seq_len=None, **kwargs):
    # modeling_bolmo.py uses rope_type="default" which was removed in newer
    # transformers; implement the standard (no-scaling) RoPE computation.
    base = getattr(config, "rope_theta", 10000.0)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, 1.0


_rope_utils.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init

_orig_init_weights = _PreTrainedModel._init_weights


def _patched_init_weights(self, module):
    # transformers._init_weights calls module.compute_default_rope_parameters
    # when rope_type=="default", but BolmoRotaryEmbedding doesn't define it.
    # Add it dynamically so the fallback lookup works.
    if (
        "RotaryEmbedding" in module.__class__.__name__
        and hasattr(module, "original_inv_freq")
        and getattr(module, "rope_type", None) == "default"
        and not hasattr(module, "compute_default_rope_parameters")
    ):
        module.compute_default_rope_parameters = lambda config: _default_rope_init(
            config
        )
    _orig_init_weights(self, module)


_PreTrainedModel._init_weights = _patched_init_weights

_orig_mlstm_post_init = _mLSTMBackendConfig.__post_init__


def _patched_mlstm_post_init(self):
    # modeling_bolmo.py hardcodes Triton kernels; replace with native equivalents
    # when CUDA is unavailable (e.g. compile-only TT systems).
    if not torch.cuda.is_available():
        triton_chunkwise = {
            "chunkwise--triton_limit_chunk",
            "chunkwise--triton_xl_chunk",
        }
        if self.chunkwise_kernel in triton_chunkwise:
            self.chunkwise_kernel = "chunkwise--native_autograd"
        if "triton" in (self.sequence_kernel or ""):
            self.sequence_kernel = "native_sequence__native"
        if self.step_kernel == "triton":
            self.step_kernel = "native"
        self.autocast_kernel_dtype = "bfloat16"
        self.inference_state_dtype = "bfloat16"
    _orig_mlstm_post_init(self)


_mLSTMBackendConfig.__post_init__ = _patched_mlstm_post_init

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
    """Available Bolmo model variants for causal language modeling."""

    BOLMO_1B = "1b"


class ModelLoader(ForgeModel):
    """Bolmo model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.BOLMO_1B: LLMModelConfig(
            pretrained_model_name="allenai/Bolmo-1B",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BOLMO_1B

    sample_text = "Language modeling is "

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
        return ModelInfo(
            model="bolmo",
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            attn_implementation="eager",
            **model_kwargs,
        )
        model.config.use_cache = False
        model.eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        prompts = [self.sample_text]

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
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.config
