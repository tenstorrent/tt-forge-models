# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron-Cascade GGUF model loader implementation for causal language modeling.

Note: The nemotron_h_moe architecture is not supported in GGUF format by transformers,
so this loader uses the base safetensors model from nvidia instead.
The nemotron_h model requires mamba-ssm (CUDA-only), so we install a pure PyTorch
stub when the real package is unavailable.
"""
import contextlib
import sys
import types
import importlib
import importlib.machinery

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


def _ensure_mamba_ssm_available():
    """Install a pure PyTorch stub for mamba_ssm if the real package is unavailable.

    The nemotron_h model requires mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn at
    import time. The real mamba-ssm package requires CUDA to install, so in CPU-only
    environments we provide a pure PyTorch fallback.
    """
    try:
        importlib.import_module("mamba_ssm.ops.triton.layernorm_gated")
        return
    except (ImportError, ModuleNotFoundError):
        pass

    def _rmsnorm_fn(
        x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=False
    ):
        dtype = x.dtype
        x = x.float()
        if group_size is not None:
            orig_shape = x.shape
            x = x.view(*x.shape[:-1], -1, group_size)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)
            x = x.view(orig_shape)
        else:
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)
        x = x.to(dtype) * weight
        if bias is not None:
            x = x + bias
        if z is not None:
            z = torch.nn.functional.silu(z)
            x = x * z
        return x

    mamba_ssm = types.ModuleType("mamba_ssm")
    mamba_ssm.__version__ = "0.0.0"
    mamba_ssm.__spec__ = importlib.machinery.ModuleSpec("mamba_ssm", None)
    mamba_ssm.ops = types.ModuleType("mamba_ssm.ops")
    mamba_ssm.ops.triton = types.ModuleType("mamba_ssm.ops.triton")

    layernorm_gated = types.ModuleType("mamba_ssm.ops.triton.layernorm_gated")
    layernorm_gated.rmsnorm_fn = _rmsnorm_fn

    selective_state = types.ModuleType("mamba_ssm.ops.triton.selective_state_update")
    selective_state.selective_state_update = None

    ssd_combined = types.ModuleType("mamba_ssm.ops.triton.ssd_combined")
    ssd_combined.mamba_chunk_scan_combined = None
    ssd_combined.mamba_split_conv1d_scan_combined = None

    mamba_ssm.ops.triton.layernorm_gated = layernorm_gated
    mamba_ssm.ops.triton.selective_state_update = selective_state
    mamba_ssm.ops.triton.ssd_combined = ssd_combined

    sys.modules["mamba_ssm"] = mamba_ssm
    sys.modules["mamba_ssm.ops"] = mamba_ssm.ops
    sys.modules["mamba_ssm.ops.triton"] = mamba_ssm.ops.triton
    sys.modules["mamba_ssm.ops.triton.layernorm_gated"] = layernorm_gated
    sys.modules["mamba_ssm.ops.triton.selective_state_update"] = selective_state
    sys.modules["mamba_ssm.ops.triton.ssd_combined"] = ssd_combined


_ensure_mamba_ssm_available()


def _patch_cuda_stream_for_cpu():
    """Patch torch.cuda.stream and default_stream to be no-ops on non-CUDA devices.

    NemotronH's forward() wraps all computation in torch.cuda.stream(), which
    raises AssertionError on CPU-only PyTorch builds. This patch makes the context
    manager a no-op when the device is not CUDA.
    """
    if torch.cuda.is_available():
        return

    _orig_default_stream = torch.cuda.default_stream
    _orig_stream = torch.cuda.stream

    def _cpu_default_stream(device=None):
        dev = (
            torch.device(device)
            if device is not None and not isinstance(device, torch.device)
            else device
        )
        if dev is None or dev.type != "cuda":
            return None
        return _orig_default_stream(device)

    @contextlib.contextmanager
    def _cpu_stream(stream):
        if stream is None:
            yield
        else:
            with _orig_stream(stream):
                yield

    torch.cuda.default_stream = _cpu_default_stream
    torch.cuda.stream = _cpu_stream


_patch_cuda_stream_for_cpu()

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
    """Available Nemotron-Cascade GGUF model variants for causal language modeling."""

    NEMOTRON_CASCADE_2_30B_A3B_MXFP4_MOE_GGUF = "Cascade_2_30B_A3B_MXFP4_MOE_GGUF"


class ModelLoader(ForgeModel):
    """Nemotron-Cascade GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_CASCADE_2_30B_A3B_MXFP4_MOE_GGUF: LLMModelConfig(
            pretrained_model_name="nvidia/Nemotron-Cascade-2-30B-A3B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_CASCADE_2_30B_A3B_MXFP4_MOE_GGUF

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
            model="Nemotron-Cascade GGUF",
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

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
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
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
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
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.config
