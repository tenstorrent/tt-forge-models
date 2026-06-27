# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.ocr model loader implementation for image-text-to-text (OCR) tasks.

dots.ocr (rednote-hilab/dots.ocr) is a multimodal OCR model: a `dots_vit`
vision transformer feeds visual tokens into a Qwen2-based causal-LM decoder
(`DotsOCRForCausalLM`). The model ships as custom HuggingFace code, so it is
loaded with `trust_remote_code=True`.

This loader drives the *full* multimodal forward pass (vision tower + text
decoder). The text decoder is also brought up on its own via the sibling
``dots_ocr/causal_lm`` loader.
"""
from typing import Optional

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available dots.ocr model variants."""

    DOTS_OCR = "dots-ocr"


class ModelLoader(ForgeModel):
    """dots.ocr model loader for image-text-to-text (OCR) tasks."""

    _VARIANTS = {
        ModelVariant.DOTS_OCR: LLMModelConfig(
            pretrained_model_name="rednote-hilab/dots.ocr",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DOTS_OCR

    # Pin the snapshot the loader was validated against. The model ships custom
    # modeling code, so a moving HEAD could change op support.
    REVISION = "c0111ce6bc07803dbc267932ffef0ae3a51dc951"

    # Short OCR-style prompt used for the sample inputs.
    sample_text = "Extract the text from the image."

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
        """Return ModelInfo for the given variant."""
        return ModelInfo(
            model="dots_ocr",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load the (custom) DotsVLProcessor for the current variant."""
        kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            revision=self.REVISION,
            **kwargs,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the dots.ocr model instance for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The DotsOCRForCausalLM multimodal model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            revision=self.REVISION,
            **model_kwargs,
        )
        model.eval()
        self.config = model.config
        return model

    @staticmethod
    def _sample_image() -> Image.Image:
        """Build a deterministic synthetic document image for OCR.

        Avoids any network dependency at test time. The image is large enough
        to satisfy the processor's ``min_pixels`` and produces a non-trivial
        number of vision patches.
        """
        img = Image.new("RGB", (476, 224), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 40), "Tenstorrent", fill=(0, 0, 0))
        draw.text((10, 90), "dots.ocr bringup", fill=(0, 0, 0))
        draw.text((10, 140), "1234567890", fill=(0, 0, 0))
        return img

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample multimodal inputs for the dots.ocr model.

        Returns a dict with ``input_ids``, ``attention_mask``, ``pixel_values``
        and ``image_grid_thw`` (Qwen2-VL style flattened patches).
        """
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image = self._sample_image()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
        # Keep only the keys the model forward consumes; the processor may emit
        # extras (e.g. mm_token_type_ids) that are not part of the graph.
        keep = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        inputs = {k: inputs[k] for k in keep if k in inputs}

        # Cast floating-point pixel values to the requested compute dtype.
        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        if batch_size != 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs
