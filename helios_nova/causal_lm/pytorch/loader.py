# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helios-Nova model loader implementation for causal language modeling.

Helios-Nova uses a custom weight naming scheme and model_type not registered
in transformers, so we implement the architecture directly in PyTorch and load
weights from safetensors.
"""

import json
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class _HeliosNovaConfig:
    vocab_size: int = 16000
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 64
    ffn_dim: int = 3072
    n_layers: int = 24
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    qk_norm: bool = True

    # Alias expected by test framework metric collection
    @property
    def num_hidden_layers(self) -> int:
        return self.n_layers

    @classmethod
    def from_dict(cls, d: dict) -> "_HeliosNovaConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # x: [batch, heads, seq, head_dim]; cos/sin: [1, 1, seq, head_dim]
    return x * cos + _rotate_half(x) * sin


class _Attention(nn.Module):
    def __init__(self, cfg: _HeliosNovaConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

        self.q_norm = _RMSNorm(cfg.head_dim, eps=cfg.norm_eps)
        self.k_norm = _RMSNorm(cfg.head_dim, eps=cfg.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK norm applied before RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Expand KV heads for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.o_proj(out)


class _FFN(nn.Module):
    def __init__(self, cfg: _HeliosNovaConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _Block(nn.Module):
    def __init__(self, cfg: _HeliosNovaConfig):
        super().__init__()
        self.attn_norm = _RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = _Attention(cfg)
        self.ffn_norm = _RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = _FFN(cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states), cos, sin)
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states


class _HeliosNovaForCausalLM(nn.Module):
    def __init__(self, cfg: _HeliosNovaConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([_Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = _RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self._build_rope_buffers(cfg)

    def _build_rope_buffers(self, cfg: _HeliosNovaConfig) -> None:
        inv_freq = 1.0 / (
            cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim)
        )
        positions = torch.arange(cfg.max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        # Shape: [1, 1, max_seq_len, head_dim]
        self.register_buffer("rope_cos", emb.cos()[None, None])
        self.register_buffer("rope_sin", emb.sin()[None, None])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        seq = input_ids.shape[1]
        hidden_states = self.tok_emb(input_ids)

        cos = self.rope_cos[:, :, :seq, :]
        sin = self.rope_sin[:, :, :seq, :]

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return CausalLMOutputWithPast(logits=logits)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class ModelVariant(StrEnum):
    """Available Helios-Nova model variants for causal language modeling."""

    HELIOS_NOVA_306M = "306M"


class ModelLoader(ForgeModel):
    """Helios-Nova model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.HELIOS_NOVA_306M: LLMModelConfig(
            pretrained_model_name="respinosamena/Helios-Nova-306M",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HELIOS_NOVA_306M

    sample_text = "What is the capital of France?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Helios-Nova",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        # Load config
        config_path = hf_hub_download(pretrained_model_name, "config.json")
        with open(config_path) as f:
            cfg_dict = json.load(f)
        model_cfg = _HeliosNovaConfig.from_dict(cfg_dict)

        if self.num_layers is not None:
            model_cfg.n_layers = self.num_layers

        # Build model
        model = _HeliosNovaForCausalLM(model_cfg)

        # Load weights; skip lm_head.weight (tied to tok_emb.weight)
        weights_path = hf_hub_download(pretrained_model_name, "model.safetensors")
        state_dict = load_file(weights_path)
        state_dict.pop("lm_head.weight", None)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # Rope buffers are computed; lm_head.weight is tied to tok_emb.weight (popped above)
        expected_missing = {"rope_cos", "rope_sin", "lm_head.weight"}
        non_expected_missing = [k for k in missing if k not in expected_missing]
        if non_expected_missing:
            raise RuntimeError(f"Unexpected missing weights: {non_expected_missing}")

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model = model.eval()
        self.config = model_cfg
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(self.sample_text, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])

    def load_config(self):
        pretrained_model_name = self._variant_config.pretrained_model_name
        config_path = hf_hub_download(pretrained_model_name, "config.json")
        with open(config_path) as f:
            cfg_dict = json.load(f)
        self.config = _HeliosNovaConfig.from_dict(cfg_dict)
        return self.config
