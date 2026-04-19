# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FIBO VLM model loader implementation for image to text.
"""

import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from typing import Optional

from ...base import ForgeModel
from ...config import (
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
    """Available FIBO VLM model variants."""

    FIBO_VLM = "fibo_vlm"


class ModelLoader(ForgeModel):
    """FIBO VLM model loader implementation for image to text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.FIBO_VLM: LLMModelConfig(
            pretrained_model_name="briaai/FIBO-vlm",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.FIBO_VLM

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

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self._precomputed = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="fibo_vlm",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_raw_inputs(self, dtype_override=None):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    @torch.no_grad()
    def _precompute_embeddings(self, full_model, raw_inputs):
        input_ids = raw_inputs["input_ids"]
        pixel_values = raw_inputs["pixel_values"]
        image_grid_thw = raw_inputs["image_grid_thw"]
        attention_mask = raw_inputs["attention_mask"]

        vl_model = full_model.model

        inputs_embeds = vl_model.get_input_embeddings()(input_ids)

        image_embeds = vl_model.get_image_features(
            pixel_values, image_grid_thw, return_dict=True
        ).pooler_output
        image_embeds = torch.cat(image_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = vl_model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        position_ids = vl_model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        return inputs_embeds, position_ids

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        full_model.eval()

        raw_inputs = self._load_raw_inputs(dtype_override=dtype_override)
        inputs_embeds, position_ids = self._precompute_embeddings(
            full_model, raw_inputs
        )
        self._precomputed = {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "attention_mask": raw_inputs["attention_mask"],
        }

        model = Wrapper(full_model)

        return model

    def load_inputs(self, dtype_override=None):
        if self._precomputed is not None:
            return self._precomputed
        raw_inputs = self._load_raw_inputs(dtype_override=dtype_override)
        return {"attention_mask": raw_inputs["attention_mask"]}
