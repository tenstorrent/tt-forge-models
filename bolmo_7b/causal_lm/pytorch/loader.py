# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bolmo-7B causal LM model loader implementation.
"""

import transformers.utils.generic as _transformers_generic
import transformers.modeling_rope_utils as _rope_utils
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional

if not hasattr(_transformers_generic, "check_model_inputs"):
    from transformers.utils.generic import merge_with_config_defaults

    def _check_model_inputs_compat(func=None):
        if func is not None:
            return merge_with_config_defaults(func)

        def decorator(f):
            return merge_with_config_defaults(f)

        return decorator

    _transformers_generic.check_model_inputs = _check_model_inputs_compat


def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **kwargs):
    rope_params = getattr(config, "rope_parameters", None)
    if isinstance(rope_params, dict):
        base = rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    else:
        base = getattr(config, "rope_theta", 10000.0)
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / head_dim
        )
    )
    return inv_freq, 1.0


if "default" not in _rope_utils.ROPE_INIT_FUNCTIONS:
    _rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

_original_init_weights = None


def _patched_init_weights(self, module):
    if (
        "RotaryEmbedding" in module.__class__.__name__
        and hasattr(module, "original_inv_freq")
        and getattr(module, "rope_type", None) == "default"
        and not hasattr(module, "compute_default_rope_parameters")
    ):
        module.compute_default_rope_parameters = _compute_default_rope_parameters
    _original_init_weights(self, module)


def _install_init_weights_patch():
    global _original_init_weights
    import transformers.modeling_utils as _modeling_utils

    if _original_init_weights is None:
        _original_init_weights = _modeling_utils.PreTrainedModel._init_weights
        _modeling_utils.PreTrainedModel._init_weights = _patched_init_weights


_install_init_weights_patch()

# Patch mLSTMBackendConfig to replace Triton kernels with native fallbacks (no CUDA on TT systems).
try:
    from xlstm.xlstm_large.model import mLSTMBackendConfig as _mLSTMBackendConfig

    _orig_mlstm_init = _mLSTMBackendConfig.__init__

    def _patched_mlstm_init(self, **kwargs):
        _TRITON_TO_NATIVE = {
            "chunkwise--triton_limit_chunk": "chunkwise--native_autograd",
            "chunkwise--triton_xl_chunk": "chunkwise--native_autograd",
            "chunkwise--triton_xl_chunk_siging": "chunkwise--native_autograd",
            "native_sequence__triton": "native_sequence__native",
            "triton": "native",
        }
        for key in ("chunkwise_kernel", "sequence_kernel", "step_kernel"):
            if key in kwargs and kwargs[key] in _TRITON_TO_NATIVE:
                kwargs[key] = _TRITON_TO_NATIVE[kwargs[key]]
        # Use bfloat16 for the native kernel to match the model dtype.
        if kwargs.get("autocast_kernel_dtype") == "float32":
            kwargs["autocast_kernel_dtype"] = "bfloat16"
        _orig_mlstm_init(self, **kwargs)

    _mLSTMBackendConfig.__init__ = _patched_mlstm_init
except ImportError:
    pass

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
    """Available Bolmo-7B model variants for causal language modeling."""

    BOLMO_7B = "7b"


class ModelLoader(ForgeModel):
    """Bolmo-7B model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.BOLMO_7B: LLMModelConfig(
            pretrained_model_name="allenai/Bolmo-7B",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BOLMO_7B

    sample_text = "Language modeling is "

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="bolmo_7b",
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
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        # The mLSTM native kernel has dtype issues with bfloat16 on non-CUDA systems;
        # use float32 to ensure consistent dtypes across all kernel operations.
        model_kwargs["torch_dtype"] = torch.float32

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
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
