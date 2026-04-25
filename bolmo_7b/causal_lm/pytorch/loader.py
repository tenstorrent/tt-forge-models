# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bolmo-7B causal LM model loader implementation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional

# The model code uses @check_model_inputs() (decorator factory) but the
# transformers compat shim only supports @check_model_inputs (plain decorator).
# Patch to support both call styles.
import transformers.utils.generic as _tfm_generic

_original_check = getattr(_tfm_generic, "check_model_inputs", None)
if _original_check is not None:

    def _check_model_inputs_patched(func=None):
        if func is None:
            return _original_check
        return _original_check(func)

    _tfm_generic.check_model_inputs = _check_model_inputs_patched

# transformers 5.6.2 omits "default" from ROPE_INIT_FUNCTIONS despite documenting it.
# Add a standard RoPE implementation (no scaling) so models using rope_type="default" work.
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE_INIT_FUNCTIONS

if "default" not in _ROPE_INIT_FUNCTIONS:

    def _compute_default_rope_parameters(
        config=None, device=None, seq_len=None, layer_type=None
    ):
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

    _ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

# BolmoXLSTMLayer hardcodes triton kernels, which require CUDA.
# Patch mLSTMBackendConfig to fall back to native PyTorch kernels when CUDA is unavailable.
if not torch.cuda.is_available():
    from mlstm_kernels.torch.backend_module import (
        mLSTMBackendConfig as _mLSTMBackendConfig,
    )

    _original_mlstm_post_init = _mLSTMBackendConfig.__post_init__

    def _mlstm_post_init_patched(self):
        if "triton" in (self.chunkwise_kernel or ""):
            self.chunkwise_kernel = "chunkwise--native_autograd"
        if "triton" in (self.sequence_kernel or ""):
            self.sequence_kernel = "native_sequence__native"
        if self.step_kernel == "triton":
            self.step_kernel = "native"
        # Match state dtype to model dtype (bfloat16) to avoid mixed-dtype
        # matmul failures when the model is loaded in bfloat16 and CUDA
        # autocast is unavailable.
        self.inference_state_dtype = "bfloat16"
        _original_mlstm_post_init(self)

    _mLSTMBackendConfig.__post_init__ = _mlstm_post_init_patched

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
