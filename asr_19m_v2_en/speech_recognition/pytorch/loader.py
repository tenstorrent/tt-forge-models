# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ASR-19M-V2-EN model loader implementation for speech recognition (ASR) using PyTorch.

abr-ai/asr-19m-v2-en is a ~19M parameter state-space model (SSM) with attention
from Applied Brain Research for English automatic speech recognition. The model
uses MFCC preprocessing, CTC loss, and SentencePiece tokenization, and ships
with custom HuggingFace auto-classes that require ``trust_remote_code=True``.
"""

from typing import Optional

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available ASR-19M-V2-EN PyTorch speech recognition model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """ASR-19M-V2-EN model loader implementation for speech recognition (PyTorch)."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="abr-ai/asr-19m-v2-en",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._feature_extractor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ASR_19M_V2_EN",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_feature_extractor(self, dtype_override=None):
        from transformers import AutoFeatureExtractor

        feature_extractor_kwargs = {}
        if dtype_override is not None:
            feature_extractor_kwargs["dtype"] = dtype_override

        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **feature_extractor_kwargs,
        )

        return self._feature_extractor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        if self._feature_extractor is None:
            self._load_feature_extractor(dtype_override=dtype_override)

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._feature_extractor(audio_array)

        return inputs
