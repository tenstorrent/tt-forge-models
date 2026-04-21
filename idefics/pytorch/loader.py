# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
IDEFICS model loader implementation for multimodal vision-text generation.
"""
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, IdeficsForVisionText2Text

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
from ...tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available IDEFICS model variants."""

    BASE_9B = "9b"


class ModelLoader(ForgeModel):
    """IDEFICS model loader for multimodal vision-text generation."""

    _VARIANTS = {
        ModelVariant.BASE_9B: ModelConfig(
            pretrained_model_name="HuggingFaceM4/idefics-9b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_9B

    sample_text = "In this picture we can see"
    sample_image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize IDEFICS model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="idefics",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the IDEFICS model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = IdeficsForVisionText2Text.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for IDEFICS."""
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        prompts = [
            [
                image,
                self.sample_text,
            ],
        ]

        inputs = self.processor(
            prompts,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs

    def decode_output(self, co_out):
        """Decode generated token ids into text."""
        if self.processor is None:
            self._load_processor()

        generated_text = self.processor.batch_decode(co_out, skip_special_tokens=True)[
            0
        ]
        print(f"Generated text: {generated_text}")
        return generated_text
