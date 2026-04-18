# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Vogent Turn model loader implementation for audio classification (turn detection).
"""

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
    """Available Vogent Turn model variants."""

    TURN_80M = "Turn_80M"


class ModelLoader(ForgeModel):
    """Vogent Turn model loader implementation for audio classification (PyTorch)."""

    _VARIANTS = {
        ModelVariant.TURN_80M: ModelConfig(
            pretrained_model_name="vogent/Vogent-Turn-80M",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TURN_80M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Vogent Turn",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        from transformers import WhisperFeatureExtractor

        self._processor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-tiny",
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Vogent Turn model instance."""
        from transformers import WhisperConfig, WhisperForAudioClassification

        config = WhisperConfig.from_pretrained("openai/whisper-tiny")
        config.num_labels = 2

        model = WhisperForAudioClassification(config)

        if dtype_override is not None:
            model.to(dtype_override)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Vogent Turn model."""
        import torch
        import numpy as np

        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        if dtype_override is not None:
            inputs = {
                k: v.to(dtype_override) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

        return inputs
