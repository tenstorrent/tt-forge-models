# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SDAR (Synergy of Diffusion and AutoRegression) model loader implementation for causal language modeling.
"""
import importlib
import os
import sys
import textwrap
import types
from pathlib import Path
from typing import Optional, TypedDict
from unittest.mock import patch

import torch
import transformers.cache_utils
import transformers.dynamic_module_utils
import transformers.modeling_rope_utils
import transformers.utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.dynamic_module_utils import (
    TRANSFORMERS_DYNAMIC_MODULE_NAME,
    _sanitize_module_name,
    get_cached_module_file,
    get_imports,
)

# transformers 5.x removed SlidingWindowCache; inject a stub so the model's remote code loads.
if not hasattr(transformers.cache_utils, "SlidingWindowCache"):
    transformers.cache_utils.SlidingWindowCache = type(
        "SlidingWindowCache", (transformers.cache_utils.Cache,), {}
    )
    sys.modules[
        "transformers.cache_utils"
    ].SlidingWindowCache = transformers.cache_utils.SlidingWindowCache

# transformers 5.2.0 dropped LossKwargs; inject a stub so the model's remote code loads.
if not hasattr(transformers.utils, "LossKwargs"):

    class LossKwargs(TypedDict, total=False):
        num_items_in_batch: int | None

    transformers.utils.LossKwargs = LossKwargs


def _default_rope_init(config, device=None, seq_len=None, **kwargs):
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    base = getattr(config, "rope_theta", 10000.0)
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


if "default" not in transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS:
    transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init


def _eager_flex_attention(
    query, key, value, attention_mask=None, scale=1.0, enable_gqa=False, **kwargs
):
    """Eager fallback replacing fused_flex_attention for non-CUDA environments."""
    num_kv_groups = query.shape[1] // key.shape[1]
    if num_kv_groups > 1:
        key = key.repeat_interleave(num_kv_groups, dim=1)
        value = value.repeat_interleave(num_kv_groups, dim=1)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale
    if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
        # Only apply if it's a 4D causal mask; 2D padding masks are not added directly.
        if attention_mask.dim() == 4:
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            attn_weights = attn_weights + causal_mask

    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)
    # Keep output in (batch, num_heads, seq_len, head_dim) to match flex_attention output format.
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, None


def _patch_sdar_module_flex_attention():
    for key, mod in sys.modules.items():
        if "modeling_sdar" in key and hasattr(mod, "fused_flex_attention"):
            mod.fused_flex_attention = _eager_flex_attention
            break


def _rms_norm_fn_stub(x, weight, bias=None, eps=1e-6, **kwargs):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    result = weight * x.to(input_dtype)
    if bias is not None:
        result = result + bias
    return result


def _flash_attn_func_stub(
    q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs
):
    # q/k/v: (batch, seqlen, num_heads, head_dim)
    num_kv_groups = q.shape[2] // k.shape[2]
    if num_kv_groups > 1:
        k = k.repeat_interleave(num_kv_groups, dim=2)
        v = v.repeat_interleave(num_kv_groups, dim=2)
    scale = softmax_scale or (q.shape[-1] ** -0.5)
    # Transpose to (batch, num_heads, seqlen, head_dim) for matmul
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    attn_weights = torch.matmul(q, k.transpose(2, 3)) * scale
    if causal:
        seqlen = q.shape[2]
        mask = torch.triu(
            torch.full((seqlen, seqlen), float("-inf"), device=q.device, dtype=q.dtype),
            diagonal=1,
        )
        attn_weights = attn_weights + mask
    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(q.dtype)
    out = torch.matmul(attn_weights, v)
    # Return (batch, seqlen, num_heads, head_dim)
    return out.transpose(1, 2).contiguous()


def _inject_flash_attn_stubs():
    if "flash_attn" not in sys.modules:
        layer_norm_mod = types.ModuleType("flash_attn.ops.triton.layer_norm")
        layer_norm_mod.rms_norm_fn = _rms_norm_fn_stub
        flash_attn_mod = types.ModuleType("flash_attn")
        flash_attn_mod.flash_attn_func = _flash_attn_func_stub
        flash_attn_mod.flash_attn_varlen_func = _flash_attn_func_stub
        sys.modules["flash_attn"] = flash_attn_mod
        sys.modules["flash_attn.ops"] = types.ModuleType("flash_attn.ops")
        sys.modules["flash_attn.ops.triton"] = types.ModuleType("flash_attn.ops.triton")
        sys.modules["flash_attn.ops.triton.layer_norm"] = layer_norm_mod
        sys.modules["flash_attn.bert_padding"] = types.ModuleType(
            "flash_attn.bert_padding"
        )


_inject_flash_attn_stubs()

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

_STUB_CONTENT = textwrap.dedent(
    """
    import torch.nn as nn

    class FusedLinearDiffusionCrossEntropyLoss(nn.Module):
        pass
"""
).lstrip()

_MISSING_MODULE_FILES = {"fused_linear_diffusion_cross_entropy.py"}


def _fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def _patched_get_cached_module_file(
    pretrained_model_name_or_path, module_file, **kwargs
):
    if os.path.basename(module_file) in _MISSING_MODULE_FILES:
        # Compute the transformers_modules path directly to avoid recursion.
        submodule = os.path.sep.join(
            map(_sanitize_module_name, pretrained_model_name_or_path.split("/"))
        )
        spec = importlib.util.find_spec(TRANSFORMERS_DYNAMIC_MODULE_NAME)
        base_dir = Path(spec.submodule_search_locations[0]) if spec else None
        if base_dir is None:
            # transformers_modules not yet created; fall through to let it fail naturally
            return get_cached_module_file(
                pretrained_model_name_or_path, module_file, **kwargs
            )
        submodule_path = base_dir / submodule
        # Find the commit-hash subdirectory (created by loading the main module file).
        for commit_dir in sorted(submodule_path.iterdir()):
            if commit_dir.is_dir():
                stub_path = commit_dir / os.path.basename(module_file)
                if not stub_path.exists():
                    stub_path.write_text(_STUB_CONTENT)
                importlib.invalidate_caches()
                return str(stub_path)
    return get_cached_module_file(pretrained_model_name_or_path, module_file, **kwargs)


class ModelVariant(StrEnum):
    """Available SDAR model variants for causal language modeling."""

    SDAR_1_7B_CHAT = "1.7B_Chat"
    SDAR_1_7B_CHAT_B32 = "1.7B_Chat_b32"


class ModelLoader(ForgeModel):
    """SDAR model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.SDAR_1_7B_CHAT: LLMModelConfig(
            pretrained_model_name="JetLM/SDAR-1.7B-Chat",
            max_length=128,
        ),
        ModelVariant.SDAR_1_7B_CHAT_B32: LLMModelConfig(
            pretrained_model_name="JetLM/SDAR-1.7B-Chat-b32",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SDAR_1_7B_CHAT

    sample_text = "Explain what reinforcement learning is in simple terms."

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
            model="SDAR",
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

        with patch(
            "transformers.dynamic_module_utils.get_imports", _fixed_get_imports
        ), patch(
            "transformers.dynamic_module_utils.get_cached_module_file",
            _patched_get_cached_module_file,
        ):
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
        if not hasattr(config, "pad_token_id"):
            config.pad_token_id = None
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers
        model_kwargs["config"] = config

        with patch(
            "transformers.dynamic_module_utils.get_imports", _fixed_get_imports
        ), patch(
            "transformers.dynamic_module_utils.get_cached_module_file",
            _patched_get_cached_module_file,
        ):
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name,
                trust_remote_code=True,
                attn_implementation="eager",
                **model_kwargs,
            ).eval()

        _patch_sdar_module_flex_attention()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
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
