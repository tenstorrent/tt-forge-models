# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UI-TARS model loader implementation for vision-language GUI agent tasks.
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
    """Available UI-TARS model variants for vision-language GUI agent tasks."""

    UI_TARS_2B_SFT = "2B_SFT"
    UI_TARS_7B_DPO = "7B_DPO"


class ModelLoader(ForgeModel):
    """UI-TARS model loader implementation for vision-language GUI agent tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.UI_TARS_2B_SFT: LLMModelConfig(
            pretrained_model_name="ByteDance-Seed/UI-TARS-2B-SFT",
        ),
        ModelVariant.UI_TARS_7B_DPO: LLMModelConfig(
            pretrained_model_name="ByteDance-Seed/UI-TARS-7B-DPO",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.UI_TARS_7B_DPO

    # Shared configuration parameters
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

    # Vision processing parameters
    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self._raw_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="UI-TARS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
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

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._raw_model = model

        return Wrapper(model)

    def load_inputs(self, dtype_override=None):
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

        from .src.model_utils import precompute_vision_inputs

        inputs_embeds, position_ids = precompute_vision_inputs(self._raw_model, inputs)

        if dtype_override is not None:
            inputs_embeds = inputs_embeds.to(dtype_override)

        return {
            "attention_mask": inputs["attention_mask"],
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
        }
