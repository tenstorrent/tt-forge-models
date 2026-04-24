# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Apexlearn BECoach-Qwen3.5-4B 4-bit VLM MLX model loader implementation for image-text-to-text generation.
"""

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
    """Available Apexlearn BECoach-Qwen3.5-4B 4-bit VLM MLX model variants."""

    BECOACH_QWEN3_5_4B_4BIT_VLM_MLX = "BECoach_Qwen3_5_4B_4bit_vlm_mlx"


class ModelLoader(ForgeModel):
    """Apexlearn BECoach-Qwen3.5-4B 4-bit VLM MLX model loader for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.BECOACH_QWEN3_5_4B_4BIT_VLM_MLX: LLMModelConfig(
            pretrained_model_name="apexlearn/BECoach-Qwen3.5-4B-4bit-vlm-mlx",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BECOACH_QWEN3_5_4B_4BIT_VLM_MLX

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Apexlearn BECoach-Qwen3.5-4B 4-bit VLM MLX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        config = AutoConfig.from_pretrained(pretrained_model_name)
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
        model_kwargs["config"] = config

        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self._variant_config.pretrained_model_name
            )

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
