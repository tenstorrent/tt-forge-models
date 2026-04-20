# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MedMO model loader implementation for image to text.
"""

import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
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
from .src.model_utils import MedMOWrapper


class ModelVariant(StrEnum):
    """Available MedMO model variants for image to text."""

    MEDMO_8B_NEXT = "8b_next"


class ModelLoader(ForgeModel):
    """MedMO model loader implementation for image to text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.MEDMO_8B_NEXT: LLMModelConfig(
            pretrained_model_name="MBZUAI/MedMO-8B-Next",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MEDMO_8B_NEXT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self._full_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MedMO",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._full_model = model

        return MedMOWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        model = self._full_model
        with torch.no_grad():
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
            pixel_values = inputs.get("pixel_values")
            image_grid_thw = inputs.get("image_grid_thw")

            inputs_embeds = model.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                pixel_values = pixel_values.type(model.model.visual.dtype)
                image_outputs = model.model.get_image_features(
                    pixel_values, image_grid_thw, return_dict=True
                )
                image_embeds = image_outputs.pooler_output
                image_embeds = torch.cat(list(image_embeds), dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                special_image_mask = input_ids == model.config.image_token_id
                special_image_mask = special_image_mask.unsqueeze(-1).expand_as(
                    inputs_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    special_image_mask, image_embeds
                )

            position_ids, _ = model.model.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
