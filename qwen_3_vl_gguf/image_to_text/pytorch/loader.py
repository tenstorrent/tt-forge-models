# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 VL GGUF model loader implementation for image to text.
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


class ModelVariant(StrEnum):
    """Available Qwen 3 VL GGUF model variants for image to text."""

    QWEN_3_VL_4B_INSTRUCT_GGUF = "4b_instruct_gguf"
    QWEN_3_VL_32B_INSTRUCT_GGUF = "32b_instruct_gguf"


class ModelLoader(ForgeModel):
    """Qwen 3 VL GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_3_VL_4B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-4B-Instruct-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_VL_32B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3-VL-32B-Instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_3_VL_4B_INSTRUCT_GGUF

    _GGUF_FILES = {
        ModelVariant.QWEN_3_VL_4B_INSTRUCT_GGUF: "Qwen3-VL-4B-Instruct-Q4_K_M.gguf",
        ModelVariant.QWEN_3_VL_32B_INSTRUCT_GGUF: "Qwen3VL-32B-Instruct-Q4_K_M.gguf",
    }

    _PROCESSOR_REPOS = {
        ModelVariant.QWEN_3_VL_4B_INSTRUCT_GGUF: "Qwen/Qwen3-VL-4B-Instruct",
        ModelVariant.QWEN_3_VL_32B_INSTRUCT_GGUF: "Qwen/Qwen3-VL-32B-Instruct",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 3 VL GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _gguf_file(self):
        return self._GGUF_FILES[self._variant]

    @property
    def _processor_repo(self):
        return self._PROCESSOR_REPOS[self._variant]

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self._gguf_file
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained(self._processor_repo)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
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
