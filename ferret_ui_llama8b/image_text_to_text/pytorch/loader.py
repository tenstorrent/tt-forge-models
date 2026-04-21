# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ferret-UI-Llama8b model loader implementation for image-text-to-text tasks.
"""

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
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
from ....tools.utils import get_file

IMAGE_TOKEN_INDEX = -200


class ModelVariant(StrEnum):
    """Available Ferret-UI-Llama8b model variants."""

    FERRET_UI_LLAMA8B = "Llama8b"


class ModelLoader(ForgeModel):
    """Ferret-UI-Llama8b model loader for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.FERRET_UI_LLAMA8B: ModelConfig(
            pretrained_model_name="jadechoghari/Ferret-UI-Llama8b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FERRET_UI_LLAMA8B

    sample_text = "Describe the image in details"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize Ferret-UI-Llama8b model loader."""
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Ferret-UI-Llama8b",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Ferret-UI-Llama8b model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Ferret-UI-Llama8b model."""
        if self.model is None:
            raise RuntimeError("load_model() must be called before load_inputs()")
        if self.tokenizer is None:
            self._load_tokenizer()

        # Build prompt with <image> placeholder; Ferret-UI substitutes it with
        # IMAGE_TOKEN_INDEX during the forward pass.
        prompt = "<image>\n" + self.sample_text

        pre, post = prompt.split("<image>", 1)
        pre_ids = self.tokenizer(
            pre, return_tensors="pt", add_special_tokens=True
        ).input_ids
        post_ids = self.tokenizer(
            post, return_tensors="pt", add_special_tokens=False
        ).input_ids
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        # Load and preprocess a sample image through the CLIP vision tower.
        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        vision_tower = self.model.get_vision_tower()
        pixel_values = vision_tower.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)
            attention_mask = attention_mask.repeat_interleave(batch_size, dim=0)
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": pixel_values,
            "image_sizes": [image.size],
        }
