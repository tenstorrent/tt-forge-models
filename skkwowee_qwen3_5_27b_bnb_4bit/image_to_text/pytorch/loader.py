# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
skkwowee/Qwen3.5-27B-bnb-4bit model loader implementation for image to text.
"""

import os

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
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


class ModelVariant(StrEnum):
    """Available skkwowee/Qwen3.5-27B-bnb-4bit model variants for image to text."""

    QWEN3_5_27B_BNB_4BIT = "27B_bnb_4bit"


class ModelLoader(ForgeModel):
    """skkwowee/Qwen3.5-27B-bnb-4bit model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_5_27B_BNB_4BIT: LLMModelConfig(
            pretrained_model_name="skkwowee/Qwen3.5-27B-bnb-4bit",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_5_27B_BNB_4BIT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="skkwowee-Qwen3.5-27B-bnb-4bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        if os.environ.get("TT_RANDOM_WEIGHTS"):
            config = AutoConfig.from_pretrained(pretrained_model_name)
            if hasattr(config, "quantization_config"):
                config.quantization_config = None
            model = AutoModelForImageTextToText.from_config(config)
            if dtype_override is not None:
                model = model.to(dtype_override)
        else:
            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs
            model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name, **model_kwargs
            )

        model.eval()
        return model

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
        return inputs
