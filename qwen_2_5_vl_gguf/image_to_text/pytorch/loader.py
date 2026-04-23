# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 2.5 VL GGUF model loader implementation for image to text.
"""

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig
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
    """Available Qwen 2.5 VL GGUF model variants for image to text."""

    QWEN_2_5_VL_72B_INSTRUCT_GGUF = "72b_instruct_gguf"
    BARTOWSKI_QWEN_2_5_VL_72B_INSTRUCT_GGUF = "bartowski_72b_instruct_gguf"


class ModelLoader(ForgeModel):
    """Qwen 2.5 VL GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_2_5_VL_72B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen2.5-VL-72B-Instruct-GGUF",
            max_length=128,
        ),
        ModelVariant.BARTOWSKI_QWEN_2_5_VL_72B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/Qwen_Qwen2.5-VL-72B-Instruct-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_2_5_VL_72B_INSTRUCT_GGUF

    # GGUF repos only carry quantized weights without full VL config/processor.
    # Load model, processor, and config from the canonical base model repo instead,
    # since transformers does not support the qwen2vl GGUF architecture.
    _BASE_MODEL_NAMES = {
        ModelVariant.QWEN_2_5_VL_72B_INSTRUCT_GGUF: "Qwen/Qwen2.5-VL-72B-Instruct",
        ModelVariant.BARTOWSKI_QWEN_2_5_VL_72B_INSTRUCT_GGUF: "Qwen/Qwen2.5-VL-72B-Instruct",
    }

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    @property
    def _base_model_name(self):
        return self._BASE_MODEL_NAMES[self._variant]

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen 2.5 VL GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(self._base_model_name)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(self._base_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._base_model_name, **model_kwargs
        ).eval()

        self.config = model.config

        if self.processor is None:
            self._load_processor()

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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(self._base_model_name)
        return self.config
