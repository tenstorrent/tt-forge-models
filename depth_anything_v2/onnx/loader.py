# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Depth Anything V2 ONNX model loader implementation for monocular depth estimation.
"""

from typing import Optional

import onnx
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Depth Anything V2 ONNX model variants."""

    SMALL = "Small"


class ModelLoader(ForgeModel):
    """Depth Anything V2 ONNX model loader implementation."""

    _VARIANTS = {
        ModelVariant.SMALL: ModelConfig(
            pretrained_model_name="onnx-community/depth-anything-v2-small",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMALL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DepthAnythingV2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Download and load the Depth Anything V2 ONNX model from Hugging Face.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_path = hf_hub_download(
            repo_id=pretrained_model_name, filename="onnx/model.onnx"
        )
        model = onnx.load(model_path)

        return model

    def load_inputs(self, **kwargs):
        """Generate sample inputs for the Depth Anything V2 ONNX model.

        Returns:
            numpy.ndarray: Preprocessed pixel values suitable for the ONNX model.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.processor(images=image, return_tensors="np")

        return inputs["pixel_values"]
