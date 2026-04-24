# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
cnfusion/Apriel-1.6-15b-Thinker-mlx-8Bit model loader for image-text-to-text tasks.
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


class ModelVariant(StrEnum):
    """Available cnfusion Apriel 1.6 15B Thinker MLX 8-bit model variants."""

    APRIEL_1_6_15B_THINKER_MLX_8BIT = "1.6-15b-Thinker-mlx-8Bit"


class ModelLoader(ForgeModel):
    """cnfusion/Apriel-1.6-15b-Thinker-mlx-8Bit model loader for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.APRIEL_1_6_15B_THINKER_MLX_8BIT: LLMModelConfig(
            pretrained_model_name="cnfusion/Apriel-1.6-15b-Thinker-mlx-8Bit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.APRIEL_1_6_15B_THINKER_MLX_8BIT

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="cnfusion Apriel 1.6 15B Thinker MLX 8-bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    # MLX-quantized weights use a packing format incompatible with standard transformers;
    # load processor and model weights from the base model in bfloat16 instead.
    _BASE_MODEL = "ServiceNow-AI/Apriel-1.6-15b-Thinker"

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(self._BASE_MODEL)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            self._BASE_MODEL, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.sample_image,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
