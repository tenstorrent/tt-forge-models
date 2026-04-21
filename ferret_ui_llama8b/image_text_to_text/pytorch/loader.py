# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ferret-UI-Llama8b model loader implementation for image-text-to-text tasks.
"""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor

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
from ....tools.utils import get_file


class ModelVariant(StrEnum):
    """Available Ferret-UI-Llama8b model variants."""

    FERRET_UI_LLAMA8B = "Llama8b"


class ModelLoader(ForgeModel):
    """Ferret-UI-Llama8b model loader for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.FERRET_UI_LLAMA8B: LLMModelConfig(
            pretrained_model_name="jadechoghari/Ferret-UI-Llama8b",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FERRET_UI_LLAMA8B

    sample_image_url = (
        "https://cdn.britannica.com/61/93061-050-99147DCE/"
        "Statue-of-Liberty-Island-New-York-Bay.jpg"
    )
    sample_text = "Describe the image in details."
    vision_tower_name = "openai/clip-vit-large-patch14-336"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.image_processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Ferret-UI-Llama8b",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def _load_image_processor(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )
        return self.image_processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()
        if self.image_processor is None:
            self._load_image_processor()

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]

        prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"<image>\n{self.sample_text}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self._variant_config.max_length,
            truncation=True,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

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
        }

    def decode_output(self, outputs, input_length=None):
        if isinstance(outputs, str):
            return outputs

        if self.tokenizer is None:
            self._load_tokenizer()

        if torch.is_tensor(outputs) and outputs.dtype in (torch.long, torch.int):
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        return self.tokenizer.decode(next_token_id, skip_special_tokens=True)
