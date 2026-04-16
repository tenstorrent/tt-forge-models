# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
IndexTTS-2 model loader implementation for text-to-speech tasks.

Builds the GPT backbone directly from ``transformers.GPT2Model`` and loads
the official IndexTeam/IndexTTS-2 checkpoint, avoiding the ``indextts``
library which is incompatible with transformers >= 5.
"""
import functools
import os

import torch
import torch.nn as nn
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


def _null_position_embeddings(range_tensor, dim):
    return torch.zeros(
        (range_tensor.shape[0], range_tensor.shape[1], dim),
        device=range_tensor.device,
    )


def _build_gpt2_backbone(model_dim, layers, heads, max_mel_tokens, max_text_tokens):
    """Reproduce the GPT-2 backbone used inside IndexTTS-2's UnifiedVoice."""
    from transformers import GPT2Config, GPT2Model

    max_mel_seq = max_mel_tokens + 2 + 1  # +2 start/stop, +1 conditioning
    max_text_seq = max_text_tokens + 2
    n_positions = max_mel_seq + max_text_seq

    gpt_config = GPT2Config(
        vocab_size=256,
        n_positions=n_positions,
        n_ctx=n_positions,
        n_embd=model_dim,
        n_layer=layers,
        n_head=heads,
        gradient_checkpointing=False,
        use_cache=True,
    )
    gpt = GPT2Model(gpt_config)
    del gpt.wpe
    gpt.wpe = functools.partial(_null_position_embeddings, dim=model_dim)
    del gpt.wte
    return gpt


class IndexTTS2GPTWrapper(nn.Module):
    """Wrapper around the IndexTTS-2 GPT-2 backbone.

    Exposes a clean forward pass that takes pre-computed input embeddings
    and produces mel token logits.
    """

    def __init__(self, transformer, final_norm, mel_head):
        super().__init__()
        self.transformer = transformer
        self.lm_head = nn.Sequential(final_norm, mel_head)

    def forward(self, inputs_embeds):
        outputs = self.transformer(inputs_embeds=inputs_embeds)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return logits


class ModelVariant(StrEnum):
    """Available IndexTTS-2 model variants."""

    INDEXTTS_2 = "IndexTTS-2"


class ModelLoader(ForgeModel):
    """IndexTTS-2 model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.INDEXTTS_2: ModelConfig(
            pretrained_model_name="IndexTeam/IndexTTS-2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INDEXTTS_2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="IndexTTS-2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from huggingface_hub import hf_hub_download

        model_dir = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="gpt.pth",
        )
        gpt_ckpt_path = model_dir

        # Config values from IndexTeam/IndexTTS-2 config.yaml
        model_dim = 1280
        layers = 24
        heads = 20
        max_mel_tokens = 1815
        max_text_tokens = 600
        number_mel_codes = 8194

        gpt = _build_gpt2_backbone(
            model_dim, layers, heads, max_mel_tokens, max_text_tokens
        )
        final_norm = nn.LayerNorm(model_dim)
        mel_head = nn.Linear(model_dim, number_mel_codes)

        # Load checkpoint and extract only the GPT-related weights
        state_dict = torch.load(gpt_ckpt_path, map_location="cpu", weights_only=True)

        gpt_state = {}
        norm_state = {}
        head_state = {}
        for key, value in state_dict.items():
            if key.startswith("gpt."):
                gpt_state[key[len("gpt.") :]] = value
            elif key.startswith("final_norm."):
                norm_state[key[len("final_norm.") :]] = value
            elif key.startswith("mel_head."):
                head_state[key[len("mel_head.") :]] = value

        gpt.load_state_dict(gpt_state, strict=False)
        final_norm.load_state_dict(norm_state)
        mel_head.load_state_dict(head_state)

        model = IndexTTS2GPTWrapper(gpt, final_norm, mel_head)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # GPT backbone hidden_size=1280, use a short sequence
        inputs_embeds = torch.randn(1, 32, 1280, dtype=dtype)
        return (inputs_embeds,)
