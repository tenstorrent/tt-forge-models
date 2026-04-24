# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher/VL-Cogito-i1-GGUF model loader for vision-language tasks.
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
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
    """Available mradermacher/VL-Cogito-i1-GGUF model variants."""

    VL_COGITO_I1_GGUF = "i1_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher/VL-Cogito-i1-GGUF model loader for vision-language tasks."""

    _VARIANTS = {
        ModelVariant.VL_COGITO_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/VL-Cogito-i1-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VL_COGITO_I1_GGUF

    # transformers does not support loading qwen2vl GGUF files natively; fall back
    # to the base fine-tune parent so we can still exercise the architecture.
    BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

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

    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="mradermacher VL-Cogito i1 GGUF",
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
            self.BASE_MODEL, **processor_kwargs
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {"low_cpu_mem_usage": True}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.BASE_MODEL, **model_kwargs
        )
        model.eval()
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

        return inputs
