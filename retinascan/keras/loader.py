# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RetinaScan model loader implementation for retinal image classification.

Supports multiple classification architectures (VGG19, ResNet50, InceptionV3,
DenseNet121, EfficientNetB0) fine-tuned on retinal scan images, hosted at
abhi1703/retinascan-models on HuggingFace.
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
class RetinaScanConfig(ModelConfig):
    """Configuration for RetinaScan variants with specific checkpoint filenames."""

    filename: str = ""
    input_size: int = 224


class ModelVariant(StrEnum):
    """Available RetinaScan model variants."""

    BEST_MODEL = "best_model"
    VGG19 = "vgg19"
    VGG19_TUNED = "vgg19_tuned"
    RESNET50 = "resnet50"
    RESNET50_TUNED = "resnet50_tuned"
    INCEPTIONV3 = "inceptionv3"
    INCEPTIONV3_TUNED = "inceptionv3_tuned"
    DENSENET121 = "densenet121"
    DENSENET121_TUNED = "densenet121_tuned"
    EFFICIENTNETB0 = "efficientnetb0"
    EFFICIENTNET_FOCAL = "efficientnet_focal"
    EFFICIENTNETB0_TUNED = "efficientnetb0_tuned"


class ModelLoader(ForgeModel):
    """RetinaScan model loader for retinal image classification tasks."""

    _REPO_ID = "abhi1703/retinascan-models"

    _VARIANTS = {
        ModelVariant.BEST_MODEL: RetinaScanConfig(
            pretrained_model_name=_REPO_ID,
            filename="best_model.keras",
        ),
        ModelVariant.VGG19: RetinaScanConfig(
            pretrained_model_name=_REPO_ID,
            filename="best_vgg19.keras",
        ),
        ModelVariant.VGG19_TUNED: RetinaScanConfig(
            pretrained_model_name=_REPO_ID,
            filename="best_vgg19_tuned.keras",
        ),
        ModelVariant.RESNET50: RetinaScanConfig(
            pretrained_model_name=_REPO_ID,
            filename="best_resnet50.keras",
        ),
        ModelVariant.RESNET50_TUNED: RetinaScanConfig(
            pretrained_model_name=_REPO_ID,
            filename="best_resnet50_tuned.keras",
        ),
        ModelVariant.INCEPTIONV3: RetinaScanConfig(
            pretrained_model_name=_REPO_ID,
            filename="best_inceptionv3.keras",
            input_size=299,
        ),
        ModelVariant.INCEPTIONV3_TUNED: RetinaScanConfig(
            pretrained_model_name=_REPO_ID,
            filename="best_inceptionv3_tuned.keras",
            input_size=299,
        ),
        ModelVariant.DENSENET121: RetinaScanConfig(
            pretrained_model_name=_REPO_ID,
            filename="best_densenet121.keras",
        ),
        ModelVariant.DENSENET121_TUNED: RetinaScanConfig(
            pretrained_model_name=_REPO_ID,
            filename="best_densenet121_tuned.keras",
        ),
        ModelVariant.EFFICIENTNETB0: RetinaScanConfig(
            pretrained_model_name=_REPO_ID,
            filename="best_efficientnetb0.keras",
        ),
        ModelVariant.EFFICIENTNET_FOCAL: RetinaScanConfig(
            pretrained_model_name=_REPO_ID,
            filename="best_efficientnet_focal.keras",
        ),
        ModelVariant.EFFICIENTNETB0_TUNED: RetinaScanConfig(
            pretrained_model_name=_REPO_ID,
            filename="best_efficientnetb0_tuned.keras",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BEST_MODEL

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
            model="RetinaScan",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.KERAS,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the RetinaScan classification model instance.

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
        """Prepare sample input for the RetinaScan classification model.

        Args:
            dtype_override: Optional numpy dtype to override the input's default dtype.
            batch_size: Number of samples in the batch. Default is 1.

        Returns:
            numpy.ndarray: Sample input tensor of shape (batch_size, H, W, 3)
            where H and W are determined by the variant's backbone input size.
        """
        size = self._variant_config.input_size
        # Keras uses channels-last format (NHWC)
        inputs = np.random.randn(batch_size, size, size, 3).astype(np.float32)

        if dtype_override is not None:
            inputs = inputs.astype(dtype_override)

        return inputs
