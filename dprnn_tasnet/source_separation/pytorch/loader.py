# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
DPRNNTasNet model loader implementation for audio source separation using PyTorch.
"""

import torch
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available DPRNNTasNet PyTorch source separation model variants."""

    KS16_WHAM_SEPCLEAN = "DPRNNTasNet-ks16_WHAM_sepclean"


class ModelLoader(ForgeModel):
    """DPRNNTasNet model loader implementation for audio source separation (PyTorch)."""

    _VARIANTS = {
        ModelVariant.KS16_WHAM_SEPCLEAN: ModelConfig(
            pretrained_model_name="julien-c/DPRNNTasNet-ks16_WHAM_sepclean",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KS16_WHAM_SEPCLEAN

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DPRNNTasNet",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from asteroid.models import BaseModel

        model = BaseModel.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        # Generate a synthetic 2-second audio waveform at 8kHz (WHAM! sample rate)
        sampling_rate = 8000
        duration_seconds = 2
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        # DPRNNTasNet expects a batch of waveforms: (batch, time)
        waveform = torch.from_numpy(audio_array).unsqueeze(0)

        if dtype_override is not None:
            waveform = waveform.to(dtype_override)

        return (waveform,)
