# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ghost-Actual Qwen3.5-4B Claude Opus 4.6 Distilled Heretic model loader
implementation for image to text.
"""

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
    """Available Ghost-Actual Qwen3.5-4B Claude Opus 4.6 Distilled Heretic variants for image to text."""

    QWEN_3_5_4B_CLAUDE_OPUS_4_6_DISTILLED_HERETIC = (
        "4b_claude_opus_4_6_distilled_heretic"
    )


class ModelLoader(ForgeModel):
    """Ghost-Actual Qwen3.5-4B Claude Opus 4.6 Distilled Heretic model loader for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_5_4B_CLAUDE_OPUS_4_6_DISTILLED_HERETIC: LLMModelConfig(
            pretrained_model_name="ghost-actual/Qwen3.5-4B-Claude-Opus-4.6-Distilled-heretic",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_4B_CLAUDE_OPUS_4_6_DISTILLED_HERETIC

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ghost_actual_qwen3_5_4b_claude_opus_4_6_distilled_heretic",
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

        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

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
        return inputs
