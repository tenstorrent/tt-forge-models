# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Descript Audio Codec (DAC) model loader implementation.
"""

import torch
import numpy as np
from transformers import DacModel, AutoProcessor
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


class ModelVariant(StrEnum):
    """Available DAC model variants."""

    DAC_24KHZ = "DAC 24kHz"


class _DacWrapper(torch.nn.Module):
    """Return only audio_values; the scalar loss in DacOutput has numel=1 and
    is undefined for PCC comparison, which would poison the minimum PCC to 0."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        return self.model(input_values).audio_values


class ModelLoader(ForgeModel):
    """Descript Audio Codec model loader implementation."""

    _VARIANTS = {
        ModelVariant.DAC_24KHZ: ModelConfig(
            pretrained_model_name="descript/dac_24khz",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DAC_24KHZ

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DAC",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DAC model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = DacModel.from_pretrained(pretrained_model_name)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return _DacWrapper(model)

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the DAC model."""
        pretrained_model_name = self._variant_config.pretrained_model_name
        processor = AutoProcessor.from_pretrained(pretrained_model_name)

        # Generate a synthetic 1-second audio waveform at 24kHz
        sampling_rate = processor.sampling_rate
        duration_seconds = 1
        audio = np.random.randn(sampling_rate * duration_seconds).astype(np.float32)

        processed = processor(
            raw_audio=audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        inputs = {"input_values": processed["input_values"]}

        if dtype_override is not None:
            inputs = {k: v.to(dtype_override) for k, v in inputs.items()}

        return inputs
