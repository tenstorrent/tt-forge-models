# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VoxCPM1.5 model loader implementation for text-to-speech tasks.
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


class VoxCPMBaseLMWrapper(nn.Module):
    """Wrapper around the VoxCPM1.5 MiniCPM4 text-semantic LM backbone.

    Exposes a clean forward pass that takes pre-computed input embeddings
    and returns the backbone hidden states.
    """

    def __init__(self, base_lm):
        super().__init__()
        self.base_lm = base_lm

    def forward(self, inputs_embeds):
        hidden_states, _ = self.base_lm(inputs_embeds, is_causal=True)
        return hidden_states


class ModelVariant(StrEnum):
    """Available VoxCPM1.5 model variants."""

    VOXCPM_1_5 = "1.5"


class ModelLoader(ForgeModel):
    """VoxCPM1.5 model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.VOXCPM_1_5: ModelConfig(
            pretrained_model_name="openbmb/VoxCPM1.5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VOXCPM_1_5

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="VoxCPM1.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from voxcpm import VoxCPM

        tts = VoxCPM.from_pretrained(
            hf_model_id=self._variant_config.pretrained_model_name,
            load_denoiser=False,
            optimize=False,
            device="cpu",
        )
        model = VoxCPMBaseLMWrapper(tts.tts_model.base_lm)
        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # MiniCPM4 backbone hidden_size=1024, use a short sequence
        inputs_embeds = torch.randn(1, 32, 1024, dtype=dtype)
        return (inputs_embeds,)
