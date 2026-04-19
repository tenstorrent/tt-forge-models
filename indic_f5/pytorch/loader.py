# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
IndicF5 model loader implementation for text-to-speech tasks.

IndicF5 is a polyglot TTS model supporting 11 Indian languages, built on the
F5-TTS Conditional Flow Matching architecture with a DiT transformer backbone.
"""
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

INDICF5_DIT_CONFIG = dict(
    dim=1024,
    depth=22,
    heads=16,
    dim_head=64,
    ff_mult=2,
    text_dim=512,
    conv_layers=4,
    text_num_embeds=2546,
    mel_dim=100,
)


class IndicF5DiTWrapper(nn.Module):
    """Wrapper around the IndicF5 DiT transformer backbone.

    Exposes a clean forward pass that takes pre-computed tensor inputs
    and produces predicted mel-spectrogram flow vectors.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, x, cond, text, time, mask):
        return self.transformer(
            x=x,
            cond=cond,
            text=text,
            time=time,
            mask=mask,
            drop_audio_cond=False,
            drop_text=False,
        )


class ModelVariant(StrEnum):
    """Available IndicF5 model variants."""

    INDIC_F5 = "IndicF5"


class ModelLoader(ForgeModel):
    """IndicF5 model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.INDIC_F5: ModelConfig(
            pretrained_model_name="6Morpheus6/IndicF5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INDIC_F5

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="IndicF5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from f5_tts.model import DiT
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        token = kwargs.pop("token", None) or os.environ.get("HF_TOKEN")
        repo_id = self._variant_config.pretrained_model_name

        transformer = DiT(**INDICF5_DIT_CONFIG)

        if not os.environ.get("TT_RANDOM_WEIGHTS"):
            weights_path = hf_hub_download(repo_id, "model.safetensors", token=token)
            state_dict = load_file(weights_path)
            prefix = "ema_model._orig_mod.transformer."
            transformer_sd = {
                k[len(prefix) :]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            transformer.load_state_dict(transformer_sd)

        if dtype_override is not None:
            transformer = transformer.to(dtype=dtype_override)

        model = IndicF5DiTWrapper(transformer)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        mel_dim = 100
        seq_len = 64
        text_len = 32
        batch = 1

        x = torch.randn(batch, seq_len, mel_dim, dtype=dtype)
        cond = torch.randn(batch, seq_len, mel_dim, dtype=dtype)
        text = torch.randint(0, 256, (batch, text_len))
        time = torch.tensor([0.5], dtype=dtype)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)

        return x, cond, text, time, mask
