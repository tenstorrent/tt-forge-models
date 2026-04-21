# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-OSS 120B EAGLE3 aimo3 speculator model loader implementation for speculative decoding.
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaMLP
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class Eagle3Attention(nn.Module):
    """EAGLE3 attention: accepts concatenated [embed, hidden] of size 2*hidden_size."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        input_size = 2 * self.hidden_size
        self.q_proj = nn.Linear(input_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            input_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            input_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(self, hidden_states):
        B, L, _ = hidden_states.shape
        q = (
            self.q_proj(hidden_states)
            .view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        # Expand k/v for grouped query attention
        num_groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(num_groups, dim=1)
        v = v.repeat_interleave(num_groups, dim=1)
        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        out = (
            out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)
        )
        return self.o_proj(out)


class Eagle3DecoderLayer(nn.Module):
    """EAGLE3 decoder layer: concatenates embed + fused hidden for attention."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = Eagle3Attention(config)
        self.mlp = LlamaMLP(config)

    def forward(self, embeds, hidden):
        # embeds: [B, L, H], hidden: [B, L, H]
        residual = hidden
        hidden = self.hidden_norm(hidden)
        embeds_normed = self.input_layernorm(embeds)
        attn_input = torch.cat([embeds_normed, hidden], dim=-1)
        attn_out = self.self_attn(attn_input)
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Eagle3Speculator(nn.Module):
    """
    EAGLE3 speculator wrapper that matches the wenliang1990/gpt-oss-120b-eagle3-aimo3
    checkpoint structure. Takes 3 concatenated verifier hidden states + input_ids.
    """

    def __init__(self, config):
        super().__init__()
        H = config.hidden_size
        num_aux = len(config.eagle_config["eagle_aux_hidden_state_layer_ids"])
        self.fc = nn.Linear(num_aux * H, H, bias=False)
        # embed_tokens not in checkpoint; use random init (verifier embeds used in practice)
        self.embed_tokens = nn.Embedding(config.vocab_size, H)
        self.midlayer = Eagle3DecoderLayer(config)
        self.norm = LlamaRMSNorm(H, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(H, config.draft_vocab_size, bias=False)
        # Vocabulary mapping buffers
        self.register_buffer(
            "d2t", torch.zeros(config.draft_vocab_size, dtype=torch.long)
        )
        self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.bool))

    def forward(self, hidden_states, input_ids):
        # hidden_states: [B, L, num_aux * H]
        fused = self.fc(hidden_states)  # [B, L, H]
        embeds = self.embed_tokens(input_ids)  # [B, L, H]
        out = self.midlayer(embeds, fused)  # [B, L, H]
        out = self.norm(out)
        return self.lm_head(out)  # [B, L, draft_vocab_size]


class ModelVariant(StrEnum):
    """Available GPT-OSS 120B EAGLE3 aimo3 speculator model variants."""

    GPT_OSS_120B_EAGLE3_AIMO3 = "120B_Eagle3_aimo3"


class ModelLoader(ForgeModel):
    """GPT-OSS 120B EAGLE3 aimo3 speculator model loader for speculative decoding.

    Loads the wenliang1990 GPT-OSS-120B EAGLE3 aimo3 speculator draft model, which
    accelerates inference of the openai/gpt-oss-120b verifier model via speculative
    decoding.
    """

    _VARIANTS = {
        ModelVariant.GPT_OSS_120B_EAGLE3_AIMO3: ModelConfig(
            pretrained_model_name="wenliang1990/gpt-oss-120b-eagle3-aimo3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPT_OSS_120B_EAGLE3_AIMO3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GPT-OSS 120B EAGLE3 aimo3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the GPT-OSS 120B EAGLE3 aimo3 speculator model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The EAGLE3 speculator model instance.
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.bfloat16

        config = AutoConfig.from_pretrained(
            cfg.pretrained_model_name, trust_remote_code=True
        )

        model = Eagle3Speculator(config)
        if dtype is not None:
            model = model.to(dtype)

        weights_path = hf_hub_download(cfg.pretrained_model_name, "model.safetensors")
        state_dict = load_file(weights_path)
        # embed_tokens not present in checkpoint; keep random init
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        expected_missing = {"embed_tokens.weight"}
        unexpected_keys = set(unexpected)
        if unexpected_keys:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected_keys}")
        if set(missing) - expected_missing:
            raise RuntimeError(
                f"Missing keys in checkpoint (beyond embed_tokens): "
                f"{set(missing) - expected_missing}"
            )

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample inputs for the EAGLE3 speculator model.

        The speculator takes concatenated hidden states from 3 verifier layers
        (hidden_size * 3 = 8640) and input token IDs.

        Args:
            dtype_override: Optional torch.dtype to override input dtype.

        Returns:
            tuple: (hidden_states, input_ids) tensors.
        """
        dtype = dtype_override or torch.bfloat16
        hidden_size = 2880  # gpt-oss-120b hidden size
        num_aux_layers = 3  # eagle_aux_hidden_state_layer_ids: [1, 17, 33]
        seq_len = 1

        torch.manual_seed(42)
        hidden_states = torch.randn(
            1, seq_len, num_aux_layers * hidden_size, dtype=dtype
        )
        input_ids = torch.randint(0, 201088, (1, seq_len))

        return hidden_states, input_ids
