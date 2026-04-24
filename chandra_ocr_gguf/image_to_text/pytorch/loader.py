# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chandra OCR GGUF model loader implementation for image to text.

Note: The qwen3vl GGUF architecture is not supported by the transformers GGUF
loader, so we load from the base Qwen3-VL-8B-Instruct safetensors checkpoint.
"""

from typing import Optional

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Chandra OCR GGUF model variants for image to text."""

    CHANDRA_OCR_Q4_K_M_GGUF = "Q4_K_M_GGUF"


class ModelLoader(ForgeModel):
    """Chandra OCR GGUF model loader implementation for image to text tasks.

    Note: Uses Qwen3-VL-8B-Instruct base weights (safetensors) because the
    qwen3vl GGUF architecture is not yet supported by transformers.
    """

    _VARIANTS = {
        ModelVariant.CHANDRA_OCR_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-8B-Instruct",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHANDRA_OCR_Q4_K_M_GGUF

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Chandra OCR GGUF",
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
                        "image": self.sample_image,
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
