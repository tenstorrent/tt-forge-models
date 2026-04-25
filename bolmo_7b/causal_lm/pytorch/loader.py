# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bolmo-7B causal LM model loader implementation.
"""

import torch
import transformers.utils.generic as _transformers_generic
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from typing import Optional

# check_model_inputs was added after transformers 5.2.0; provide a no-op stub
if not hasattr(_transformers_generic, "check_model_inputs"):

    def _check_model_inputs(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    _transformers_generic.check_model_inputs = _check_model_inputs

# "default" rope type and compute_default_rope_parameters were added after transformers 5.2.0.
# Provide a basic RoPE implementation (standard inv_freq, no scaling) for compatibility.
if "default" not in ROPE_INIT_FUNCTIONS:

    def _compute_default_rope_parameters(config=None, device=None, **kwargs):
        base = getattr(config, "rope_theta", 10000.0)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
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

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    # transformers 5.2.0 _init_weights uses module.compute_default_rope_parameters for rope_type=="default";
    # patch it to fall back to ROPE_INIT_FUNCTIONS["default"] when the module doesn't have that method.
    import transformers.modeling_utils as _modeling_utils

    _original_init_weights = _modeling_utils.PreTrainedModel._init_weights

    def _patched_init_weights(self, module):
        if (
            "RotaryEmbedding" in module.__class__.__name__
            and hasattr(module, "original_inv_freq")
            and getattr(module, "rope_type", None) == "default"
            and not hasattr(module, "compute_default_rope_parameters")
        ):
            module.compute_default_rope_parameters = lambda cfg: ROPE_INIT_FUNCTIONS[
                "default"
            ](cfg)
        _original_init_weights(self, module)

    _modeling_utils.PreTrainedModel._init_weights = _patched_init_weights

# Bolmo hardcodes triton-based mlstm kernels; fall back to native (CPU) kernels when CUDA is absent.
if not torch.cuda.is_available():
    from mlstm_kernels.torch.backend_module import (
        mLSTMBackendConfig as _mLSTMBackendConfig,
    )

    _original_mlstm_backend_init = _mLSTMBackendConfig.__init__

    def _patched_mlstm_backend_init(self, **kwargs):
        if "triton" in kwargs.get("chunkwise_kernel", ""):
            kwargs["chunkwise_kernel"] = "chunkwise--native_autograd"
        if "triton" in kwargs.get("sequence_kernel", ""):
            kwargs["sequence_kernel"] = "native_sequence__native"
        if kwargs.get("step_kernel") == "triton":
            kwargs["step_kernel"] = "native"
        # Use bfloat16 for all kernel dtypes; float32 inference_state_dtype causes dtype mismatch
        # in the TT-torch matmul override when states mix with bfloat16 weight tensors.
        kwargs["autocast_kernel_dtype"] = "bfloat16"
        kwargs["inference_state_dtype"] = "bfloat16"
        _original_mlstm_backend_init(self, **kwargs)

    _mLSTMBackendConfig.__init__ = _patched_mlstm_backend_init

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

        # Bolmo processes tokens at byte-level then patches them; the tokenizer's
        # attention_mask is byte-length while h_patch is shorter, causing a shape
        # mismatch in create_sliding_window_causal_mask.  Drop it so the model
        # computes causal masking internally.
        inputs.pop("attention_mask", None)

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
