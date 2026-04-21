# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LungScan-BD model loader implementation for lung scan classification.

Uses a ConvNeXt Base backbone trained on lung scan imagery for tuberculosis
detection.
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
class LungScanBDConfig(ModelConfig):
    """Configuration for LungScan-BD model variants with checkpoint filenames."""

    filename: str = ""


class ModelVariant(StrEnum):
    """Available LungScan-BD model variants."""

    CONVNEXT_BASE_TB = "ConvNeXtBase_TB"


class ModelLoader(ForgeModel):
    """LungScan-BD model loader for lung scan classification tasks."""

    _VARIANTS = {
        ModelVariant.CONVNEXT_BASE_TB: LungScanBDConfig(
            pretrained_model_name="nesaruddin3227/lungscan-bd-models",
            filename="models/ConvNeXtBase_TB.keras",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONVNEXT_BASE_TB

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
            model="LungScan-BD",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.KERAS,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LungScan-BD classification model instance.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            keras.Model: The loaded Keras classification model.
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
        """Prepare sample input for the LungScan-BD classification model.

        Args:
            dtype_override: Optional numpy dtype to override the input's default dtype.
            batch_size: Number of samples in the batch. Default is 1.

        Returns:
            numpy.ndarray: Sample input tensor of shape (batch_size, 224, 224, 3)
        """
        # Keras uses channels-last format (NHWC); ConvNeXt Base uses 224x224 inputs.
        inputs = np.random.randn(batch_size, 224, 224, 3).astype(np.float32)

        if dtype_override is not None:
            inputs = inputs.astype(dtype_override)

        return inputs
