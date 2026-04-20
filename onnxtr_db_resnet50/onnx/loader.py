# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OnnxTR DB ResNet50 model loader implementation for text detection.

This model is a Differentiable Binarization (DB) text detection model with a
ResNet50 backbone, exported to ONNX format as part of the OnnxTR project.
"""
import numpy as np
import onnx
import torch
from PIL import Image
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
    """Available OnnxTR DB ResNet50 model variants."""

    DB_RESNET50 = "db-resnet50"


class ModelLoader(ForgeModel):
    """OnnxTR DB ResNet50 model loader for text detection."""

    _VARIANTS = {
        ModelVariant.DB_RESNET50: ModelConfig(
            pretrained_model_name="Felix92/onnxtr-db-resnet50",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DB_RESNET50

    # Model input normalization parameters from config.json
    INPUT_HEIGHT = 1024
    INPUT_WIDTH = 1024
    MEAN = [0.798, 0.785, 0.772]
    STD = [0.264, 0.2749, 0.287]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="onnxtr_db_resnet50",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the ONNX model from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download

        pretrained = self._variant_config.pretrained_model_name
        model_path = hf_hub_download(pretrained, "model.onnx")
        model = onnx.load(model_path)

        return model

    def load_inputs(self, *, dtype_override=None, batch_size=1):
        """Prepare sample input for the DB ResNet50 text detection model.

        Creates a synthetic document-like image normalized with the model's
        expected mean and std values. Input shape: [batch, 3, 1024, 1024].
        """
        image = Image.new(
            "RGB", (self.INPUT_WIDTH, self.INPUT_HEIGHT), color=(200, 200, 200)
        )

        img_array = np.array(image, dtype=np.float32) / 255.0
        # Normalize with model-specific mean and std
        mean = np.array(self.MEAN, dtype=np.float32)
        std = np.array(self.STD, dtype=np.float32)
        img_array = (img_array - mean) / std
        # HWC -> CHW
        img_array = img_array.transpose(2, 0, 1)

        inputs = torch.from_numpy(img_array).unsqueeze(0)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        inputs = inputs.repeat_interleave(batch_size, dim=0)

        return inputs
