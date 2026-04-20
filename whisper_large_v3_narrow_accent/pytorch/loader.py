# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Whisper-Large v3 Narrow Accent classifier loader.

Loads ``tiantiaf/whisper-large-v3-narrow-accent``, a LoRA fine-tuned
Whisper-Large v3 encoder with a lightweight convolutional head that
classifies speaker audio into one of 16 narrow English accent classes.
"""

from typing import Optional

import torch

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
from .src.model_utils import WhisperWrapper


class ModelVariant(StrEnum):
    """Available Whisper-Large v3 narrow-accent classifier variants."""

    NARROW_ACCENT = "Narrow_Accent"


class ModelLoader(ForgeModel):
    """Loader for the Vox-Profile narrow-accent Whisper classifier."""

    _VARIANTS = {
        ModelVariant.NARROW_ACCENT: ModelConfig(
            pretrained_model_name="tiantiaf/whisper-large-v3-narrow-accent",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NARROW_ACCENT

    # Whisper expects 16 kHz mono audio and the classifier is trained on
    # 3-15 second clips; a 3-second sample is sufficient for bring-up.
    SAMPLING_RATE = 16000
    SAMPLE_DURATION_SECONDS = 3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model: Optional[WhisperWrapper] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WhisperLargeV3NarrowAccent",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model = WhisperWrapper.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        model.eval()
        if dtype_override is not None:
            model = model.to(dtype_override)
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        num_samples = self.SAMPLING_RATE * self.SAMPLE_DURATION_SECONDS
        audio = torch.zeros((1, num_samples), dtype=torch.float32)
        if dtype_override is not None:
            audio = audio.to(dtype_override)
        return [audio]
