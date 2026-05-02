# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-heretic-i1-GGUF model loader for causal language modeling.
"""

import importlib.util
import sys
import types
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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

# The GGUF file uses architecture 'nemotron_h_moe' which is not supported by
# transformers' GGUF loader. NemotronH also requires mamba_ssm (CUDA library)
# at import time. We stub out mamba_ssm and load config/tokenizer from the
# official NVIDIA repo, then initialise the model with random weights.
# Random weights are sufficient: both CPU and TT paths use the same weights,
# so PCC exercises compiler correctness rather than weight quality.
_BASE_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def _rmsnorm_fn(
    x,
    weight,
    bias=None,
    z=None,
    eps=1e-5,
    group_size=None,
    norm_before_gate=True,
    **kwargs,
):
    """Pure-PyTorch gated group RMS norm (drop-in for mamba_ssm rmsnorm_fn).

    With norm_before_gate=False (Mamba2 SSD convention):
      output = group_rms_norm(x * silu(z)) * weight
    With norm_before_gate=True:
      output = group_rms_norm(x) * weight * silu(z)
    """
    import torch.nn.functional as F

    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    orig_dtype = x.dtype
    x = x.float()
    if group_size is not None and group_size > 0 and x.shape[-1] % group_size == 0:
        orig_shape = x.shape
        x = x.reshape(*orig_shape[:-1], -1, group_size)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + eps)
        x = x.reshape(orig_shape)
    else:
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + eps)
    if weight is not None:
        x = x * weight.float()
    if bias is not None:
        x = x + bias.float()
    x = x.to(orig_dtype)
    if z is not None and norm_before_gate:
        x = x * F.silu(z.to(orig_dtype))
    return x


def _install_mamba_ssm_stub():
    """Inject a minimal mamba_ssm stub so NemotronH can import without CUDA.

    Only mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn is required at import
    time. The Triton SSM kernels are guarded by is_mamba_2_ssm_available() which
    returns False on non-CUDA systems, so they are never called.
    """
    if "mamba_ssm" in sys.modules:
        return

    def _mod(name):
        m = types.ModuleType(name)
        m.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        sys.modules[name] = m
        return m

    mamba = _mod("mamba_ssm")
    ops = _mod("mamba_ssm.ops")
    triton = _mod("mamba_ssm.ops.triton")
    lg = _mod("mamba_ssm.ops.triton.layernorm_gated")
    lg.rmsnorm_fn = _rmsnorm_fn
    _mod("mamba_ssm.ops.triton.ssd_combined")
    _mod("mamba_ssm.ops.triton.selective_state_update")

    mamba.ops = ops
    ops.triton = triton
    triton.layernorm_gated = lg


_install_mamba_ssm_stub()


class ModelVariant(StrEnum):
    """Available mradermacher/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-heretic-i1-GGUF model variants."""

    NVIDIA_NEMOTRON_3_NANO_30B_A3B_BF16_HERETIC_I1_GGUF = (
        "NVIDIA_Nemotron_3_Nano_30B_A3B_BF16_heretic_i1_GGUF"
    )


class ModelLoader(ForgeModel):
    """mradermacher/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-heretic-i1-GGUF model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.NVIDIA_NEMOTRON_3_NANO_30B_A3B_BF16_HERETIC_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-heretic-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NVIDIA_NEMOTRON_3_NANO_30B_A3B_BF16_HERETIC_I1_GGUF

    GGUF_FILE = "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-heretic.i1-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="mradermacher/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-heretic-i1-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            _BASE_MODEL_NAME, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.config is None:
            self.load_config()

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        config = self.config
        if self.num_layers is not None:
            from copy import deepcopy

            config = deepcopy(self.config)
            if hasattr(config, "layers_block_type"):
                config.layers_block_type = config.layers_block_type[: self.num_layers]
            config.num_hidden_layers = self.num_layers

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        if self.tokenizer.chat_template is not None:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = self.sample_text
        prompts = [text]

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
            _BASE_MODEL_NAME, trust_remote_code=True
        )
        return self.config
