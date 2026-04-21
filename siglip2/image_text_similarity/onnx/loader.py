# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SigLIP2 ONNX model loader implementation for image-text similarity.
"""

from typing import Optional

import onnx
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor

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
    """Available SigLIP2 ONNX model variants."""

    BASE_PATCH16_256 = "Base_Patch16_256"


class ModelLoader(ForgeModel):
    """SigLIP2 ONNX model loader implementation for image-text similarity."""

    _VARIANTS = {
        ModelVariant.BASE_PATCH16_256: ModelConfig(
            pretrained_model_name="onnx-community/siglip2-base-patch16-256-ONNX",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_PATCH16_256

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SigLIP2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Download and load the SigLIP2 ONNX model from Hugging Face.

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
        """Generate sample image and text inputs for the SigLIP2 ONNX model.

        Returns:
            dict: Input arrays (pixel_values, input_ids) for the ONNX model.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]
        text_prompts = ["a photo of 2 cats", "a photo of 2 dogs"]

        inputs = self.processor(
            text=text_prompts,
            images=image,
            return_tensors="np",
            padding="max_length",
        )

        return {
            "pixel_values": inputs["pixel_values"],
            "input_ids": inputs["input_ids"],
        }
