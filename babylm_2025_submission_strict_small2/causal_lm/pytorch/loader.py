# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BabyLM 2025 Submission strict-small2 model loader implementation for causal language modeling.

A BabyLM 2025 strict-small track submission based on a custom xQwen architecture
that blends Qwen-style MLPs with xLSTM attention. The repo ships remote code
(modeling_xqwen / configuration_xqwen), so the model is loaded with
trust_remote_code=True via AutoModelForCausalLM.
"""
import contextlib
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **kwargs):
    """Standard RoPE without scaling, for compatibility with transformers 5.x which removed 'default'."""
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    rope_theta = getattr(config, "rope_theta", 10000.0)
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    attention_scaling = 1.0
    return inv_freq, attention_scaling


if "default" not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _patch_fla_cpu():
    """Patch fla ShortConvolution to use native PyTorch ops on CPU (fla-core 0.5.0 has no CPU path)."""
    try:
        import torch.nn.functional as F
        from fla.modules.conv.short_conv import ShortConvolution

        def _cpu_forward(
            self,
            x,
            residual=None,
            mask=None,
            cache=None,
            output_final_state=False,
            **kwargs
        ):
            # x: [B, T, D] -> transpose to [B, D, T] for conv1d
            x_t = x.transpose(1, 2)
            # weight shape: [D, 1, K] for depthwise
            weight = self.weight
            bias = self.bias
            # causal padding: pad left with kernel_size-1 zeros, trim right padding
            x_padded = F.pad(x_t, (self.kernel_size[0] - 1, 0))
            y_t = F.conv1d(x_padded, weight, bias, groups=self.groups)
            if mask is not None:
                y_t = y_t * mask.transpose(1, 2)
            if self.activation in ("silu", "swish"):
                y_t = torch.nn.functional.silu(y_t)
            y = y_t.transpose(1, 2)
            if residual is not None:
                y = y + residual
            final_state = None
            return y, final_state

        if not torch.cuda.is_available():
            ShortConvolution.forward = _cpu_forward
    except ImportError:
        pass


_patch_fla_cpu()


class ModelVariant(StrEnum):
    """Available BabyLM 2025 Submission strict-small2 model variants."""

    STRICT_SMALL2 = "strict-small2"


class ModelLoader(ForgeModel):
    """BabyLM 2025 Submission strict-small2 loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.STRICT_SMALL2: LLMModelConfig(
            pretrained_model_name="PatrickHaller/babylm_2025_submission_strict-small2",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STRICT_SMALL2

    sample_text = "The child played with"

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
            model="BabyLM 2025 Submission strict-small2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if not hasattr(config, "pad_token_id"):
            config.pad_token_id = None
        # Use native (non-Triton) backends for CPU compatibility
        if not torch.cuda.is_available():
            config.chunkwise_kernel = "chunkwise--native_autograd"
            config.sequence_kernel = "native_sequence__native"
            config.step_kernel = "native"
            config.inference_state_dtype = "bfloat16"
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        predicted_token_ids = logits.argmax(dim=-1)
        predicted_text = self.tokenizer.decode(
            predicted_token_ids[0], skip_special_tokens=True
        )

        return predicted_text
