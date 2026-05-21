# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HuBERT model loader implementation for audio classification (emotion recognition).
"""

from typing import Optional

import numpy as np
import torch
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

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
    """Available HuBERT audio-classification variants."""

    LARGE_SUPERB_ER = "Large_Superb_Er"


class ModelLoader(ForgeModel):
    """HuBERT model loader implementation for audio classification."""

    _VARIANTS = {
        ModelVariant.LARGE_SUPERB_ER: ModelConfig(
            pretrained_model_name="superb/hubert-large-superb-er",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_SUPERB_ER

    # HuBERT (and the underlying wav2vec2 feature extractor) expects 16 kHz audio.
    sampling_rate = 16000

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._feature_extractor = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Return model info for dashboard/metrics reporting."""
        return ModelInfo(
            model="HuBERT",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_feature_extractor(self):
        """Load and cache the Wav2Vec2 feature extractor used by the model card."""
        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self._model_name
        )
        return self._feature_extractor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the HuBERT sequence-classification model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The HuBERT model instance.
        """
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = HubertForSequenceClassification.from_pretrained(
            self._model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the HuBERT audio classification model.

        Generates a deterministic 1-second 16 kHz dummy audio waveform and runs
        it through the Wav2Vec2 feature extractor that the model card specifies.

        Args:
            dtype_override: Optional torch.dtype to cast input features to.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self._feature_extractor is None:
            self._load_feature_extractor()

        # Deterministic 1-second waveform at 16 kHz. Keep amplitude small so the
        # normalization the feature extractor applies stays well-conditioned.
        rng = np.random.default_rng(0)
        audio = rng.standard_normal(self.sampling_rate).astype(np.float32) * 0.1

        inputs = self._feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            inputs["input_values"] = inputs["input_values"].to(dtype_override)

        return dict(inputs)
