# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Typhoon OCR model loader implementation for image-to-text OCR tasks.
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
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
from .src.model import Wrapper
from .src.model_utils import precompute_inputs_embeds


class ModelVariant(StrEnum):
    """Available Typhoon OCR model variants for image-to-text tasks."""

    TYPHOON_OCR_3B = "3B"


class ModelLoader(ForgeModel):
    """Typhoon OCR model loader implementation for image-to-text OCR tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.TYPHOON_OCR_3B: LLMModelConfig(
            pretrained_model_name="typhoon-ai/typhoon-ocr-3b",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.TYPHOON_OCR_3B

    # Shared configuration parameters
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                },
                {
                    "type": "text",
                    "text": (
                        "Below is an image of a document page along with its dimensions. "
                        "Simply return the markdown representation of this document, presenting tables in markdown format as they naturally appear.\n"
                        "If the document contains images, use a placeholder like dummy.png for each image.\n"
                        "Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
                        "RAW_TEXT_START\n\nRAW_TEXT_END"
                    ),
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
        self._hf_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Typhoon OCR",
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

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True, "use_cache": False}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._hf_model = model
        model = Wrapper(model)

        return model

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

        return precompute_inputs_embeds(self._hf_model, inputs)
