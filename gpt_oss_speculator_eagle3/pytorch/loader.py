# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

HIDDEN_SIZE = 2880
NUM_ATTENTION_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 64
INTERMEDIATE_SIZE = 2880
DRAFT_VOCAB_SIZE = 64000
TARGET_VOCAB_SIZE = 201088
RMS_NORM_EPS = 1e-05
ROPE_THETA = 10000.0


class Eagle3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary_emb(x, cos, sin):
    return (x * cos) + (_rotate_half(x) * sin)


def _build_rope(seq_len, dtype, device):
    inv_freq = 1.0 / (
        ROPE_THETA
        ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=device) / HEAD_DIM)
    )
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return (
        emb.cos().to(dtype).unsqueeze(0).unsqueeze(0),
        emb.sin().to(dtype).unsqueeze(0).unsqueeze(0),
    )


class Eagle3Attention(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 2 * HIDDEN_SIZE
        self.q_proj = nn.Linear(input_size, NUM_ATTENTION_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(input_size, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(input_size, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_ATTENTION_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)

    def forward(self, hidden_states, cos, sin):
        bsz, q_len, _ = hidden_states.size()
        q = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, NUM_ATTENTION_HEADS, HEAD_DIM)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, NUM_KV_HEADS, HEAD_DIM)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, NUM_KV_HEADS, HEAD_DIM)
            .transpose(1, 2)
        )

        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        num_groups = NUM_ATTENTION_HEADS // NUM_KV_HEADS
        k = (
            k.unsqueeze(2)
            .expand(-1, -1, num_groups, -1, -1)
            .reshape(bsz, NUM_ATTENTION_HEADS, q_len, HEAD_DIM)
        )
        v = (
            v.unsqueeze(2)
            .expand(-1, -1, num_groups, -1, -1)
            .reshape(bsz, NUM_ATTENTION_HEADS, q_len, HEAD_DIM)
        )

        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(HEAD_DIM)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = (
            torch.matmul(attn, v)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, q_len, NUM_ATTENTION_HEADS * HEAD_DIM)
        )
        return self.o_proj(out)


class Eagle3MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Eagle3DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = Eagle3RMSNorm(HIDDEN_SIZE, RMS_NORM_EPS)
        self.hidden_norm = Eagle3RMSNorm(HIDDEN_SIZE, RMS_NORM_EPS)
        self.post_attention_layernorm = Eagle3RMSNorm(HIDDEN_SIZE, RMS_NORM_EPS)
        self.self_attn = Eagle3Attention()
        self.mlp = Eagle3MLP()

    def forward(self, hidden_states, cos, sin):
        embeds = hidden_states[:, :, :HIDDEN_SIZE]
        hidden = hidden_states[:, :, HIDDEN_SIZE:]

        hidden = self.hidden_norm(hidden)
        residual = hidden

        embeds = self.input_layernorm(embeds)
        attn_input = torch.cat([embeds, hidden], dim=-1)
        hidden_states = residual + self.self_attn(attn_input, cos, sin)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Eagle3DraftModelStandalone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(TARGET_VOCAB_SIZE, HIDDEN_SIZE)
        self.input_norm = Eagle3RMSNorm(3 * HIDDEN_SIZE, RMS_NORM_EPS)
        self.fc = nn.Linear(3 * HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.layers = nn.ModuleList([Eagle3DecoderLayer()])
        self.norm = Eagle3RMSNorm(HIDDEN_SIZE, RMS_NORM_EPS)
        self.lm_head = nn.Linear(HIDDEN_SIZE, DRAFT_VOCAB_SIZE, bias=False)
        self.register_buffer("d2t", torch.zeros(DRAFT_VOCAB_SIZE, dtype=torch.long))
        self.register_buffer("t2d", torch.zeros(TARGET_VOCAB_SIZE, dtype=torch.bool))

    def forward(self, hidden_states, input_ids):
        embeds = self.embed_tokens(input_ids)
        hidden_states = self.input_norm(hidden_states)
        fused = self.fc(hidden_states)
        combined = torch.cat([embeds, fused], dim=-1)
        cos, sin = _build_rope(combined.shape[1], combined.dtype, combined.device)
        for layer in self.layers:
            combined = layer(combined, cos, sin)
        out = self.norm(combined)
        return self.lm_head(out)


class ModelVariant(StrEnum):
    GPT_OSS_20B_EAGLE3 = "20B_Eagle3"


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.GPT_OSS_20B_EAGLE3: ModelConfig(
            pretrained_model_name="RedHatAI/gpt-oss-20b-speculator.eagle3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPT_OSS_20B_EAGLE3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GPT-OSS Speculator EAGLE3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        cfg = self._variant_config
        dtype = dtype_override or torch.bfloat16

        model = Eagle3DraftModelStandalone()

        weights_path = hf_hub_download(cfg.pretrained_model_name, "model.safetensors")
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(dtype=dtype)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.bfloat16
        hidden_size = 2880
        seq_len = 1

        torch.manual_seed(42)
        hidden_states = torch.randn(1, seq_len, 3 * hidden_size, dtype=dtype)
        input_ids = torch.randint(0, TARGET_VOCAB_SIZE, (1, seq_len))

        return {"hidden_states": hidden_states, "input_ids": input_ids}
