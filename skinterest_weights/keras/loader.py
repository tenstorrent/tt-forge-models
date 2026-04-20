# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Skinterest Weights model loader implementation for skin image classification.

Source: https://huggingface.co/aishasalimg/skinterest-weights

The repository hosts several Keras classifiers bundled together:
- A Kaggle-trained skin classifier.
- An undertone classifier.
- Three "merged12" fine-tuned classifiers using EfficientNet, ResNet50V2, and
  Xception backbones.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
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
class SkinterestWeightsConfig(ModelConfig):
    """Configuration for Skinterest Weights variants with checkpoint filenames."""

    filename: str = ""
    input_shape: Tuple[int, int, int] = (224, 224, 3)


class ModelVariant(StrEnum):
    """Available Skinterest Weights model variants."""

    KAGGLE_CLASSIFIER = "KaggleClassifier"
    UNDERTONE_CLASSIFIER = "UndertoneClassifier"
    MERGED12_EFFNET = "Merged12EfficientNet"
    MERGED12_RESNET50V2 = "Merged12ResNet50V2"
    MERGED12_XCEPTION = "Merged12Xception"


class ModelLoader(ForgeModel):
    """Skinterest Weights Keras model loader for skin image classification."""

    _VARIANTS = {
        ModelVariant.KAGGLE_CLASSIFIER: SkinterestWeightsConfig(
            pretrained_model_name="aishasalimg/skinterest-weights",
            filename="kaggle_classifier_model.keras",
            input_shape=(224, 224, 3),
        ),
        ModelVariant.UNDERTONE_CLASSIFIER: SkinterestWeightsConfig(
            pretrained_model_name="aishasalimg/skinterest-weights",
            filename="undertone_classifier_model.keras",
            input_shape=(224, 224, 3),
        ),
        ModelVariant.MERGED12_EFFNET: SkinterestWeightsConfig(
            pretrained_model_name="aishasalimg/skinterest-weights",
            filename="merged12_effnet_custom_finetuned.keras",
            input_shape=(224, 224, 3),
        ),
        ModelVariant.MERGED12_RESNET50V2: SkinterestWeightsConfig(
            pretrained_model_name="aishasalimg/skinterest-weights",
            filename="merged12_resnet50v2_finetuned.keras",
            input_shape=(224, 224, 3),
        ),
        ModelVariant.MERGED12_XCEPTION: SkinterestWeightsConfig(
            pretrained_model_name="aishasalimg/skinterest-weights",
            filename="merged12_xception_finetuned.keras",
            input_shape=(299, 299, 3),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MERGED12_EFFNET

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Skinterest Weights",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.KERAS,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import keras

        config = self._variant_config
        model_path = hf_hub_download(
            repo_id=config.pretrained_model_name,
            filename=config.filename,
        )
        model = keras.models.load_model(model_path)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        # Keras image classifiers use channels-last (NHWC) layout.
        h, w, c = self._variant_config.input_shape
        inputs = np.random.randn(batch_size, h, w, c).astype(np.float32)

        if dtype_override is not None:
            inputs = inputs.astype(dtype_override)

        return inputs
