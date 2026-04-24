# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Evo-1 model loader implementation for causal language modeling.
"""
import math
import sys
import types

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional

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


def _inject_flash_attn_stub():
    """Inject minimal CPU-compatible flash_attn stubs if flash_attn is not installed.

    The evo model uses flash_attn's MHA and RotaryEmbedding classes. On CPU-only
    systems, flash_attn is not installable, so we provide stubs with the same
    interface that allow model instantiation and tracing.
    """
    if "flash_attn" in sys.modules:
        return

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class RotaryEmbedding(nn.Module):
        def __init__(
            self,
            dim,
            base=10000.0,
            interleaved=False,
            scale_base=None,
            pos_idx_in_fp32=True,
            device=None,
        ):
            super().__init__()
            self.dim = dim
            self.base = float(base)
            self.interleaved = interleaved
            self.scale_base = scale_base
            self.pos_idx_in_fp32 = pos_idx_in_fp32
            self.scale = None
            self._seq_len_cached = 0
            self._cos_cached = None
            self._sin_cached = None
            inv_freq = 1.0 / (
                self.base
                ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
            )
            self.register_buffer("inv_freq", inv_freq)

        def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
            if seqlen > self._seq_len_cached or self._cos_cached is None:
                self._seq_len_cached = seqlen
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                freqs = torch.outer(t, self.inv_freq.float())
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)

        def forward(self, qkv, seqlen_offset=0, **kwargs):
            seqlen = qkv.shape[1]
            self._update_cos_sin_cache(
                seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype
            )
            return qkv

    class MHA(nn.Module):
        def __init__(
            self,
            embed_dim,
            num_heads,
            num_heads_kv=None,
            rotary_emb_dim=0,
            qkv_proj_bias=True,
            rotary_emb_base=10000,
            causal=False,
            layer_idx=None,
            out_proj_bias=True,
            use_flash_attn=False,
            cross_attn=False,
            **kwargs,
        ):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
            self.head_dim = embed_dim // num_heads
            self.causal = causal
            self.cross_attn = cross_attn
            self.layer_idx = layer_idx

            total_kv_dim = (num_heads + 2 * self.num_heads_kv) * self.head_dim
            self.Wqkv = nn.Linear(embed_dim, total_kv_dim, bias=qkv_proj_bias)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias)

            if rotary_emb_dim > 0:
                self.rotary_emb = RotaryEmbedding(
                    dim=rotary_emb_dim,
                    base=rotary_emb_base,
                )
            else:
                self.rotary_emb = None

        def forward(self, x, inference_params=None, **kwargs):
            B, L, C = x.shape
            qkv = self.Wqkv(x)
            q_dim = self.num_heads * self.head_dim
            kv_dim = self.num_heads_kv * self.head_dim
            q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
            q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(B, L, self.num_heads_kv, self.head_dim).transpose(1, 2)
            v = v.reshape(B, L, self.num_heads_kv, self.head_dim).transpose(1, 2)
            scale = math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) / scale
            if self.causal:
                mask = torch.triu(
                    torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
                )
                attn = attn.masked_fill(mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, L, self.embed_dim)
            return self.out_proj(out)

    # Build the stub package hierarchy and inject into sys.modules
    flash_attn = types.ModuleType("flash_attn")
    flash_attn_layers = types.ModuleType("flash_attn.layers")
    flash_attn_layers_rotary = types.ModuleType("flash_attn.layers.rotary")
    flash_attn_layers_rotary.RotaryEmbedding = RotaryEmbedding
    flash_attn_modules = types.ModuleType("flash_attn.modules")
    flash_attn_modules_mha = types.ModuleType("flash_attn.modules.mha")
    flash_attn_modules_mha.MHA = MHA

    flash_attn.layers = flash_attn_layers
    flash_attn.modules = flash_attn_modules
    flash_attn_layers.rotary = flash_attn_layers_rotary
    flash_attn_modules.mha = flash_attn_modules_mha

    sys.modules["flash_attn"] = flash_attn
    sys.modules["flash_attn.layers"] = flash_attn_layers
    sys.modules["flash_attn.layers.rotary"] = flash_attn_layers_rotary
    sys.modules["flash_attn.modules"] = flash_attn_modules
    sys.modules["flash_attn.modules.mha"] = flash_attn_modules_mha


class ModelVariant(StrEnum):
    """Available Evo model variants for causal language modeling."""

    EVO_1_8K_BASE = "1-8k-base"
    EVO_1_131K_BASE = "1-131k-base"


class ModelLoader(ForgeModel):
    """Evo-1 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.EVO_1_8K_BASE: LLMModelConfig(
            pretrained_model_name="togethercomputer/evo-1-8k-base",
            max_length=256,
        ),
        ModelVariant.EVO_1_131K_BASE: LLMModelConfig(
            pretrained_model_name="togethercomputer/evo-1-131k-base",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EVO_1_8K_BASE

    sample_text = "ACGTACGTACGTACGTACGTACGTACGTACGT"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

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
            model="Evo",
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

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                # ByteTokenizer has no special tokens; use NUL byte as pad
                self.tokenizer.add_special_tokens({"pad_token": "\x00"})
                self.tokenizer.pad_token_id = 0

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Evo-1 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Evo-1 model instance for causal language modeling.
        """
        _inject_flash_attn_stub()

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name,
                trust_remote_code=True,
                revision="1.1_fix",
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            revision="1.1_fix",
            **model_kwargs,
        )

        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Evo-1 model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        prompts = [self.sample_text] * batch_size

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        return inputs
