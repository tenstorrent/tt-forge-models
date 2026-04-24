# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Firdhokk Speech Emotion Recognition (Wav2Vec2 XLSR-53) model loader for audio classification.
"""

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
    """Available Firdhokk Speech Emotion Recognition model variants."""

    XLSR_53 = "XLSR_53"


class ModelLoader(ForgeModel):
    """Firdhokk Speech Emotion Recognition model loader (PyTorch)."""

    _VARIANTS = {
        ModelVariant.XLSR_53: ModelConfig(
            pretrained_model_name="firdhokk/speech-emotion-recognition-with-facebook-wav2vec2-large-xlsr-53",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XLSR_53

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="FirdhokkSpeechEmotionRecognition",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        from transformers import AutoFeatureExtractor

        processor_kwargs = {"do_normalize": True, "return_attention_mask": True}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        self._processor = AutoFeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForAudioClassification

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForAudioClassification.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import torch
        import numpy as np

        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Generate a synthetic 1-second audio waveform at 16kHz
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
