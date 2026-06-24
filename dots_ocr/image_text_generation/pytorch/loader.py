# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.ocr model loader implementation for image-to-text (document OCR) tasks.

dots.ocr (rednote-hilab/dots.ocr) is a vision-language OCR model whose language
backbone is a Qwen2 decoder (``DotsOCRForCausalLM`` subclasses ``Qwen2ForCausalLM``)
and whose vision tower (``dots_vit``) is a Qwen2-VL-style NaViT encoder. The
checkpoint ships custom modeling code, so it is loaded with ``trust_remote_code``.
"""
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model import Wrapper


class ModelVariant(StrEnum):
    """Available dots.ocr model variants."""

    DOTS_OCR = "base"


class ModelLoader(ForgeModel):
    """dots.ocr model loader implementation for image-to-text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DOTS_OCR: LLMModelConfig(
            pretrained_model_name="rednote-hilab/dots.ocr",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DOTS_OCR

    # Shared configuration parameters
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                },
                {"type": "text", "text": "Extract the text from this image."},
            ],
        }
    ]

    # Vision processing parameters - kept modest so the vision sequence (and the
    # resulting on-device footprint) stays small.
    min_pixels = 56 * 56
    max_pixels = 4 * 28 * 28 * 64

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="dots_ocr",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the dots.ocr model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The wrapped dots.ocr model instance for image-to-text tasks.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.config.use_cache = False
        self.config = model.config
        model.eval()
        model = Wrapper(model)
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the dots.ocr model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
                           If specified, converts pixel_values to the specified dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        # Apply chat template to get the text prompt
        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(self.messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Keep only the inputs the wrapper consumes, in a stable order.
        inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
