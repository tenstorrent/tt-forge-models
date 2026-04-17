# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 27B bf16 model loader implementation for image to text.
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
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


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds, attention_mask, position_ids):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return outputs.logits


class ModelVariant(StrEnum):
    """Available Qwen 3.5 27B bf16 model variants for image to text."""

    QWEN_3_5_27B_BF16 = "27b_bf16"


class ModelLoader(ForgeModel):
    """Qwen 3.5 27B bf16 model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_27B_BF16: LLMModelConfig(
            pretrained_model_name="mlx-community/Qwen3.5-27B-bf16",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_27B_BF16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self._full_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3.5 27B bf16",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        target_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        model = model.to(target_dtype)

        self._full_model = model

        return Wrapper(model)

    def load_inputs(self, dtype_override=None):
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

        with torch.no_grad():
            inner_model = self._full_model.model
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            inputs_embeds = inner_model.get_input_embeddings()(input_ids)

            if "pixel_values" in inputs and inputs["pixel_values"] is not None:
                image_grid_thw = inputs.get("image_grid_thw", None)
                image_outputs = inner_model.get_image_features(
                    inputs["pixel_values"], image_grid_thw, return_dict=True
                )
                image_embeds = image_outputs.pooler_output
                image_embeds = torch.cat(image_embeds, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                image_mask, _ = inner_model.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    image_features=image_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            position_ids = inner_model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=inputs.get("image_grid_thw", None),
                video_grid_thw=inputs.get("video_grid_thw", None),
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
