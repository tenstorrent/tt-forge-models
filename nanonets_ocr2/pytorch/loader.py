# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nanonets OCR2 model loader implementation for document OCR tasks.
"""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
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
    """Available Nanonets OCR2 model variants for document OCR tasks."""

    NANONETS_OCR2_1_5B_EXP = "1.5B_Exp"


class ModelLoader(ForgeModel):
    """Nanonets OCR2 model loader implementation for document OCR tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.NANONETS_OCR2_1_5B_EXP: LLMModelConfig(
            pretrained_model_name="nanonets/Nanonets-OCR2-1.5B-exp",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.NANONETS_OCR2_1_5B_EXP

    # Shared configuration parameters
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {
                    "type": "text",
                    "text": "Convert the document to markdown.",
                },
            ],
        }
    ]

    # Vision processing parameters
    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self._precomputed = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Nanonets OCR2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        processor_kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self.processor

    def _load_raw_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor()

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

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    @torch.no_grad()
    def _precompute_embeddings(self, full_model, raw_inputs):
        from .src.model_utils import apply_patches

        apply_patches()

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

        full_model = Qwen2VLForConditionalGeneration.from_pretrained(
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
