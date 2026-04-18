# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SAM3 (Segment Anything Model 3) loader implementation
"""

import torch
from typing import Optional
from PIL import Image
from loguru import logger
from transformers import Sam3Config, Sam3Model, Sam3Processor

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from datasets import load_dataset


def _build_processor_locally():
    """Build Sam3Processor from local components (facebook/sam3 is a gated repo)."""
    from transformers import CLIPTokenizerFast, Sam3ImageProcessorFast

    image_processor = Sam3ImageProcessorFast()
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    return Sam3Processor(image_processor=image_processor, tokenizer=tokenizer)


class ModelVariant(StrEnum):
    """Available SAM3 model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """SAM3 model loader implementation."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="facebook/sam3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SAM3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_processor(self, model_name, **kwargs):
        try:
            return Sam3Processor.from_pretrained(model_name, **kwargs)
        except OSError:
            return _build_processor_locally()

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        if "config" not in kwargs:
            kwargs["config"] = Sam3Config()
        kwargs.setdefault("attn_implementation", "eager")

        framework_model = Sam3Model.from_pretrained(model_name, **kwargs).to("cpu")

        self.processor = self._get_processor(model_name)

        if dtype_override is not None:
            framework_model = framework_model.to(dtype_override)

        return framework_model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            model_name = self._variant_config.pretrained_model_name
            self.processor = self._get_processor(model_name)

        try:
            dataset = load_dataset("huggingface/cats-image")["test"]
            raw_image = dataset[0]["image"].convert("RGB")
        except Exception as e:
            logger.warning(
                f"Failed to load image from dataset. Using random fallback tensor. Reason: {e}"
            )
            raw_image = Image.fromarray(
                (torch.rand(3, 1024, 1024) * 255).byte().permute(1, 2, 0).numpy()
            )

        # Use text prompt for SAM3 segmentation
        inputs = self.processor(images=raw_image, text="cat", return_tensors="pt").to(
            "cpu"
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
