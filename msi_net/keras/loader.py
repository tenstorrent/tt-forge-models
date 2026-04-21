# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MSI-Net loader implementation for visual saliency prediction.

MSI-Net is a contextual encoder-decoder network with a VGG16 backbone and an
ASPP (Atrous Spatial Pyramid Pooling) module. It predicts where humans fixate
on natural images, producing a per-pixel saliency map.
"""

import numpy as np
from typing import Optional
from huggingface_hub import snapshot_download

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
    """Available MSI-Net model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """MSI-Net loader for visual saliency prediction."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="alexanderkroner/MSI-Net",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # MSI-Net accepts (320, 320), (240, 320), or (320, 240) depending on the
    # aspect ratio of the input image; use the square shape for synthetic inputs.
    input_height = 320
    input_width = 320

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
            model="MSI-Net",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.KERAS,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MSI-Net saliency prediction model.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            keras.Model: The loaded Keras saliency prediction model.
        """
        import tensorflow as tf

        # MSI-Net is distributed as a TensorFlow SavedModel directory rather
        # than a single .keras/.h5 file, so the whole repo snapshot is needed.
        config = self._variant_config
        model_dir = snapshot_download(repo_id=config.pretrained_model_name)
        model = tf.keras.models.load_model(model_dir)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample input for the MSI-Net saliency prediction model.

        Args:
            dtype_override: Optional numpy dtype to override the input's default dtype.
            batch_size: Number of samples in the batch. Default is 1.

        Returns:
            numpy.ndarray: Sample input tensor of shape
                (batch_size, input_height, input_width, 3).
        """
        # Keras uses channels-last format (NHWC); MSI-Net expects float32 pixel
        # values in the 0-255 range (see preprocess_input in the model card).
        inputs = np.random.uniform(
            low=0.0,
            high=255.0,
            size=(batch_size, self.input_height, self.input_width, 3),
        ).astype(np.float32)

        if dtype_override is not None:
            inputs = inputs.astype(dtype_override)

        return inputs
