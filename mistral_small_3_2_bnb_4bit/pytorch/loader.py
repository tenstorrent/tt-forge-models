# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral Small 3.2 BNB 4-bit model loader implementation for image-text-to-text.
"""

from typing import Optional

import torch
from transformers import AutoConfig, AutoProcessor, Mistral3ForConditionalGeneration

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
    """Available Mistral Small 3.2 BNB 4-bit model variants."""

    MISTRAL_SMALL_3_2_24B_INSTRUCT_2506_BNB_4BIT = (
        "Small_3.2_24B_Instruct_2506_BNB_4bit"
    )


class ModelLoader(ForgeModel):
    """Mistral Small 3.2 BNB 4-bit model loader for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.MISTRAL_SMALL_3_2_24B_INSTRUCT_2506_BNB_4BIT: ModelConfig(
            pretrained_model_name="unsloth/Mistral-Small-3.2-24B-Instruct-2506-bnb-4bit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MISTRAL_SMALL_3_2_24B_INSTRUCT_2506_BNB_4BIT

    sample_text = "What action do you think I should take in this situation? List all the possible actions and explain why you think they are good or bad."
    sample_image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Mistral-Small-3.2-BNB-4bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {"device_map": "cpu"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = Mistral3ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        from PIL import Image

        from ...tools.utils import cast_input_to_type, get_file

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        text_prompt = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.sample_text},
                    ],
                }
            ],
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt",
        )

        if dtype_override is not None:
            if "pixel_values" in inputs:
                inputs["pixel_values"] = cast_input_to_type(
                    inputs["pixel_values"], dtype_override
                )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
