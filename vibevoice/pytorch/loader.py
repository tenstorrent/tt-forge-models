# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VibeVoice model loader implementation for text-to-speech tasks.
"""

from typing import Optional

import torch
import torch.nn as nn

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


class VibeVoiceQwen2Wrapper(nn.Module):
    """Wrapper around the VibeVoice Qwen2 LLM backbone.

    Exposes a clean forward pass that takes pre-computed input embeddings
    and produces hidden states from the Qwen2 decoder.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds):
        outputs = self.model(inputs_embeds=inputs_embeds, use_cache=False)
        return outputs.last_hidden_state


class ModelVariant(StrEnum):
    """Available VibeVoice model variants."""

    VIBEVOICE_7B_BNB_4BIT = "7B-bnb-4bit"


class ModelLoader(ForgeModel):
    """VibeVoice model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.VIBEVOICE_7B_BNB_4BIT: ModelConfig(
            pretrained_model_name="marksverdhai/vibevoice-7b-bnb-4bit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIBEVOICE_7B_BNB_4BIT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="VibeVoice",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from huggingface_hub import hf_hub_download
        from transformers import Qwen2Config, Qwen2Model
        import json

        # The checkpoint stores BnB 4-bit quantized weights that require
        # bitsandbytes to deserialize. Since we only compile the Qwen2
        # backbone, instantiate it directly from the decoder config.
        config_path = hf_hub_download(
            self._variant_config.pretrained_model_name, "config.json"
        )
        with open(config_path) as f:
            full_config = json.load(f)

        qwen2_config = Qwen2Config(**full_config["decoder_config"])
        qwen2 = Qwen2Model(qwen2_config)
        if dtype_override is not None:
            qwen2 = qwen2.to(dtype_override)

        model = VibeVoiceQwen2Wrapper(qwen2)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.bfloat16
        # Qwen2 backbone hidden_size=3584, use a short sequence of embeddings
        inputs_embeds = torch.randn(1, 32, 3584, dtype=dtype)
        return (inputs_embeds,)
