# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ASID Captioner 7B i1 GGUF model loader implementation for image to text.
"""

from PIL import Image
from transformers import (
    Qwen2VLConfig,
    Qwen2VLForConditionalGeneration,
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
    """Available ASID Captioner 7B i1 GGUF model variants for image to text."""

    ASID_CAPTIONER_7B_I1_Q4_K_M_GGUF = "7b_i1_Q4_K_M_gguf"


class ModelLoader(ForgeModel):
    """ASID Captioner 7B i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.ASID_CAPTIONER_7B_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/ASID-Captioner-7B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ASID_CAPTIONER_7B_I1_Q4_K_M_GGUF

    BASE_MODEL = "AudioVisual-Caption/ASID-Captioner-7B"
    GGUF_FILE = "ASID-Captioner-7B.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ASID Captioner 7B i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL)

        # qwen2vl GGUF architecture is not yet supported by the transformers GGUF loader;
        # load config from base model and initialize with random weights
        config = Qwen2VLConfig.from_pretrained(self.BASE_MODEL)
        model = Qwen2VLForConditionalGeneration(config)
        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.new("RGB", (224, 224)),
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
