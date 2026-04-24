# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BabyLM 2025 Submission Strict model loader implementation for causal language modeling.

A 0.3B-parameter BabyLM 2025 strict-small track submission using an xqwen-based
architecture with Flash Linear Attention (FLA) and MLSTM kernels.
Source: https://huggingface.co/PatrickHaller/babylm_2025_submission_strict
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from typing import Optional


def _default_rope_init(config, device=None, seq_len=None, layer_type=None, **kwargs):
    base = getattr(config, "rope_theta", 10000.0)
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, 1.0


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
    """Available BabyLM 2025 Submission Strict model variants."""

    BABYLM_2025_SUBMISSION_STRICT = "babylm_2025_submission_strict"


class ModelLoader(ForgeModel):
    """BabyLM 2025 Submission Strict model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.BABYLM_2025_SUBMISSION_STRICT: LLMModelConfig(
            pretrained_model_name="PatrickHaller/babylm_2025_submission_strict",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BABYLM_2025_SUBMISSION_STRICT

    sample_text = "Once upon a time"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BabyLM 2025 Submission Strict",
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if "default" not in ROPE_INIT_FUNCTIONS:
            ROPE_INIT_FUNCTIONS["default"] = _default_rope_init

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = self.tokenizer.pad_token_id
        # Disable sliding window attention: uses flex_attention which requires CUDA
        config.use_sliding_window = False
        config.sliding_window = None
        # Override GPU-only triton kernels with CPU-compatible native kernels
        config.chunkwise_kernel = "chunkwise--native_autograd"
        config.sequence_kernel = "native_sequence__native"
        config.step_kernel = "native"
        # Disable float32 norm reductions to keep all tensors in bfloat16
        config.norm_reduction_force_float32 = False
        # Use train mode: config.json has mode="inference" with float32 states,
        # which causes dtype mismatch (float32 state tensors mixed with bfloat16 Q/K/V)
        config.mode = "train"
        # return_last_states must be True: model always unpacks (h, state) from backend
        config.return_last_states = True
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
        )

        return inputs
