# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MAI-UI-8B i1 GGUF model loader implementation for image to text.

Note: The qwen3vl GGUF architecture is not yet supported by the transformers
GGUF loader, so we load from the HF-native base checkpoint instead.
"""

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
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

BASE_MODEL = "Tongyi-MAI/MAI-UI-8B"


class ModelVariant(StrEnum):
    """Available MAI-UI-8B i1 GGUF model variants for image to text."""

    MAI_UI_8B_I1_GGUF = "mai_ui_8b_i1_gguf"
    MAI_UI_8B_GGUF = "mai_ui_8b_gguf"


class ModelLoader(ForgeModel):
    """MAI-UI-8B i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.MAI_UI_8B_I1_GGUF: LLMModelConfig(
            pretrained_model_name=BASE_MODEL,
            max_length=128,
        ),
        ModelVariant.MAI_UI_8B_GGUF: LLMModelConfig(
            pretrained_model_name=BASE_MODEL,
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MAI_UI_8B_I1_GGUF

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MAI-UI-8B i1 GGUF",
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

        model = Qwen3VLForConditionalGeneration.from_pretrained(
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
