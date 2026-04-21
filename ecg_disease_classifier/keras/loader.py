# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ECG Disease Classifier loader implementation for pediatric cardiac condition detection.

An Enhanced 1D CNN with Squeeze-Excitation blocks and temporal attention that
performs multi-label classification of 19 cardiac conditions from variable-length
12-lead pediatric ECG signals sampled at 500 Hz.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from huggingface_hub import hf_hub_download

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


@dataclass
class EcgDiseaseClassifierConfig(ModelConfig):
    """Configuration for ECG Disease Classifier variants with checkpoint filename."""

    filename: str = ""


class ModelVariant(StrEnum):
    """Available ECG Disease Classifier variants."""

    FINAL = "final"


class ModelLoader(ForgeModel):
    """ECG Disease Classifier loader for pediatric cardiac condition detection."""

    _VARIANTS = {
        ModelVariant.FINAL: EcgDiseaseClassifierConfig(
            pretrained_model_name="Neural-Network-Project/ECG-Disease-Classifier",
            filename="archive/final_model.keras",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FINAL

    # 12-lead ECG sampled at 500 Hz; pick a 10-second window (5000 samples)
    # within the model's 5-112 second training range.
    _SAMPLING_RATE = 500
    _SIGNAL_SECONDS = 10
    _NUM_CHANNELS = 12

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ECG Disease Classifier",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.KERAS,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ECG Disease Classifier Keras model.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            keras.Model: The loaded Keras multi-label ECG classifier.
        """
        import keras

        config = self._variant_config
        model_path = hf_hub_download(
            repo_id=config.pretrained_model_name,
            filename=config.filename,
        )
        model = keras.models.load_model(model_path)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample 12-lead ECG input for the classifier.

        Args:
            dtype_override: Optional numpy dtype to override the input's default dtype.
            batch_size: Number of samples in the batch. Default is 1.

        Returns:
            numpy.ndarray: Sample ECG tensor of shape
                (batch_size, sampling_rate * signal_seconds, num_channels).
        """
        # Keras 1D CNN uses channels-last format: (batch, timesteps, channels).
        timesteps = self._SAMPLING_RATE * self._SIGNAL_SECONDS
        inputs = np.random.randn(batch_size, timesteps, self._NUM_CHANNELS).astype(
            np.float32
        )

        if dtype_override is not None:
            inputs = inputs.astype(dtype_override)

        return inputs
