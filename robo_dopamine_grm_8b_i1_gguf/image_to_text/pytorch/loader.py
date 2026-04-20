# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Robo-Dopamine GRM 8B i1 GGUF model loader implementation for image to text.
"""
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
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
    """Available Robo-Dopamine GRM 8B i1 GGUF model variants for image to text."""

    ROBO_DOPAMINE_GRM_8B_I1_GGUF = "8b_i1_gguf"


class ModelLoader(ForgeModel):
    """Robo-Dopamine GRM 8B i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.ROBO_DOPAMINE_GRM_8B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/Robo-Dopamine-GRM-8B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ROBO_DOPAMINE_GRM_8B_I1_GGUF

    GGUF_FILE = "Robo-Dopamine-GRM-8B.i1-Q4_K_M.gguf"

    # Processor source (the GGUF repo ships only quantized weights).
    _PROCESSOR_SOURCE = "tanhuajie2001/Robo-Dopamine-GRM-8B"

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Robo-Dopamine GRM 8B i1 GGUF",
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
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(self._PROCESSOR_SOURCE)

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
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

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
