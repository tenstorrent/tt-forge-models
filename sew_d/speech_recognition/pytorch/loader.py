# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
SEW-D (Squeezed and Efficient Wav2vec with DeBERTa) model loader for speech recognition (ASR).
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
    """Available SEW-D speech recognition model variants."""

    TINY_100K_FT_LS100H = "Tiny_100k_ft_ls100h"


class ModelLoader(ForgeModel):
    """SEW-D model loader implementation for speech recognition (PyTorch)."""

    _VARIANTS = {
        ModelVariant.TINY_100K_FT_LS100H: ModelConfig(
            pretrained_model_name="asapp/sew-d-tiny-100k-ft-ls100h",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_100K_FT_LS100H

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SEW-D",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import Wav2Vec2Processor

        self._processor = Wav2Vec2Processor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self._processor

    def load_model(self, **kwargs):
        from transformers import SEWDForCTC

        model = SEWDForCTC.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        model.eval()

        return model

    def load_inputs(self):
        import numpy as np

        if self._processor is None:
            self._load_processor()

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

        return inputs
