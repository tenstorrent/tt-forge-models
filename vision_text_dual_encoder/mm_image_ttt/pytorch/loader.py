# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VisionTextDualEncoder model loader implementation for image-text tasks.

Mirrors the Habana/clip GaudiConfig usage pattern from the
Optimum Habana run_clip example, combining a CLIP vision encoder with a
RoBERTa text encoder via ``VisionTextDualEncoderModel``.
"""
import torch
from typing import Optional

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
    """Available VisionTextDualEncoder model variants for image-text tasks."""

    CLIP_VIT_LARGE_PATCH14_ROBERTA_LARGE = "CLIP_ViT_Large_Patch14_RoBERTa_Large"


class ModelLoader(ForgeModel):
    """VisionTextDualEncoder model loader implementation for image-text tasks."""

    _VARIANTS = {
        ModelVariant.CLIP_VIT_LARGE_PATCH14_ROBERTA_LARGE: ModelConfig(
            pretrained_model_name="openai/clip-vit-large-patch14",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CLIP_VIT_LARGE_PATCH14_ROBERTA_LARGE

    _VISION_TEXT_MODEL_PAIRS = {
        ModelVariant.CLIP_VIT_LARGE_PATCH14_ROBERTA_LARGE: (
            "openai/clip-vit-large-patch14",
            "roberta-large",
        ),
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Vision Text Dual Encoder",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _vision_and_text_paths(self):
        return self._VISION_TEXT_MODEL_PAIRS[self._variant]

    def _load_processor(self):
        from transformers import (
            AutoImageProcessor,
            AutoTokenizer,
            VisionTextDualEncoderProcessor,
        )

        vision_path, text_path = self._vision_and_text_paths()
        image_processor = AutoImageProcessor.from_pretrained(vision_path)
        tokenizer = AutoTokenizer.from_pretrained(text_path)
        self.processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the VisionTextDualEncoder model instance for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The VisionTextDualEncoder model instance.
        """
        from transformers import VisionTextDualEncoderModel

        vision_path, text_path = self._vision_and_text_paths()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = VisionTextDualEncoderModel.from_vision_text_pretrained(
            vision_path, text_path, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the VisionTextDualEncoder model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        from datasets import load_dataset

        if self.processor is None:
            self._load_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        inputs = self.processor(
            text=["a photo of a cat", "a photo of a dog"],
            images=image,
            return_tensors="pt",
            padding=True,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
