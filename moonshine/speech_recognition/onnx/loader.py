# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Moonshine ONNX model loader implementation for speech recognition (ASR).
"""

from typing import Optional

import numpy as np
import onnx
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor

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
    """Available Moonshine ONNX speech recognition model variants."""

    TINY = "Tiny"


class ModelLoader(ForgeModel):
    """Moonshine ONNX model loader implementation for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.TINY: ModelConfig(
            pretrained_model_name="onnx-community/moonshine-tiny-ONNX",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Moonshine",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the Moonshine ONNX encoder model."""
        encoder_path = hf_hub_download(
            self._variant_config.pretrained_model_name,
            filename="onnx/encoder_model.onnx",
        )
        model = onnx.load(encoder_path)
        return model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the Moonshine ONNX encoder model."""
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self._variant_config.pretrained_model_name,
            )

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = self._processor.feature_extractor.sampling_rate
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="np",
        )

        return inputs.input_values
