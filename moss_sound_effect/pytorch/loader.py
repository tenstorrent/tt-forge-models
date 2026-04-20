# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MOSS-SoundEffect model loader implementation for text-to-audio generation.
"""
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


class MossSoundEffectWrapper(nn.Module):
    """Wrapper around the MOSS-SoundEffect backbone.

    Exposes a clean forward pass through the MossTTSDelay language backbone,
    producing hidden states suitable for audio code prediction.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.lm_head = model.lm_head

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return logits


class ModelVariant(StrEnum):
    """Available MOSS-SoundEffect model variants."""

    MOSS_SOUND_EFFECT = "moss-sound-effect"


class ModelLoader(ForgeModel):
    """MOSS-SoundEffect model loader for text-to-audio generation."""

    _VARIANTS = {
        ModelVariant.MOSS_SOUND_EFFECT: ModelConfig(
            pretrained_model_name="OpenMOSS-Team/MOSS-SoundEffect",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOSS_SOUND_EFFECT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MOSS-SoundEffect",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        full_model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=dtype_override or torch.float32,
            **kwargs,
        )
        model = MossSoundEffectWrapper(full_model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        # Simulate tokenized input: batch of 1, short sequence
        input_ids = torch.randint(0, 1000, (1, 32))
        attention_mask = torch.ones(1, 32, dtype=torch.long)
        return (input_ids, attention_mask)
