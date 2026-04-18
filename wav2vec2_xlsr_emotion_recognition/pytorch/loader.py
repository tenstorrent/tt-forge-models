# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wav2Vec2 XLSR Speech Emotion Recognition model loader for audio classification.
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
    """Available Wav2Vec2 XLSR emotion recognition model variants."""

    XLSR_EN = "XLSR_EN"
    RF_EN = "RF_EN"


class ModelLoader(ForgeModel):
    """Wav2Vec2 XLSR Speech Emotion Recognition model loader (PyTorch)."""

    _VARIANTS = {
        ModelVariant.XLSR_EN: ModelConfig(
            pretrained_model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        ),
        ModelVariant.RF_EN: ModelConfig(
            pretrained_model_name="r-f/wav2vec-english-speech-emotion-recognition",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XLSR_EN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Wav2Vec2XLSREmotionRecognition",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        from transformers import Wav2Vec2FeatureExtractor

        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        import torch
        from transformers import Wav2Vec2ForSequenceClassification

        model_kwargs = {"torch_dtype": dtype_override or torch.float32}
        model_kwargs |= kwargs

        dtype = dtype_override or torch.float32
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        model.to(dtype)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np
        import torch

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

        # Only pass input_values; attention_mask is all-ones for a single
        # unpadded sample and its int32 dtype triggers an S64-vs-S32 XLA error.
        inputs = {"input_values": inputs["input_values"]}

        if dtype_override is not None:
            inputs = {
                k: (
                    v.to(dtype_override)
                    if isinstance(v, torch.Tensor) and v.is_floating_point()
                    else v
                )
                for k, v in inputs.items()
            }

        return inputs
