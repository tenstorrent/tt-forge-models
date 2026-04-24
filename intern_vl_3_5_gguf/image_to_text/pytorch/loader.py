# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InternVL3.5 GGUF model loader implementation for image to text.
"""

from transformers import (
    AutoConfig,
    AutoProcessor,
    InternVLForConditionalGeneration,
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
    """Available InternVL3.5 GGUF model variants for image to text."""

    INTERN_VL3_5_4B_Q4_K_M = "4b_q4_k_m"
    INTERN_VL3_5_4B_Q8_0 = "4b_q8_0"
    INTERN_VL3_5_14B_Q4_K_M = "14b_q4_k_m"
    INTERN_VL3_5_14B_Q8_0 = "14b_q8_0"


class ModelLoader(ForgeModel):
    """InternVL3.5 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.INTERN_VL3_5_4B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="bartowski/OpenGVLab_InternVL3_5-4B-GGUF",
            max_length=128,
        ),
        ModelVariant.INTERN_VL3_5_4B_Q8_0: LLMModelConfig(
            pretrained_model_name="bartowski/OpenGVLab_InternVL3_5-4B-GGUF",
            max_length=128,
        ),
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="bartowski/OpenGVLab_InternVL3_5-14B-GGUF",
            max_length=128,
        ),
        ModelVariant.INTERN_VL3_5_14B_Q8_0: LLMModelConfig(
            pretrained_model_name="bartowski/OpenGVLab_InternVL3_5-14B-GGUF",
            max_length=128,
        ),
    }

    _HF_PROCESSORS = {
        ModelVariant.INTERN_VL3_5_4B_Q4_K_M: "OpenGVLab/InternVL3_5-4B-HF",
        ModelVariant.INTERN_VL3_5_4B_Q8_0: "OpenGVLab/InternVL3_5-4B-HF",
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: "OpenGVLab/InternVL3_5-14B-HF",
        ModelVariant.INTERN_VL3_5_14B_Q8_0: "OpenGVLab/InternVL3_5-14B-HF",
    }

    # HF model names for loading configs (GGUF files only contain text backbone)
    _HF_CONFIGS = {
        ModelVariant.INTERN_VL3_5_4B_Q4_K_M: "OpenGVLab/InternVL3_5-4B-HF",
        ModelVariant.INTERN_VL3_5_4B_Q8_0: "OpenGVLab/InternVL3_5-4B-HF",
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: "OpenGVLab/InternVL3_5-14B-HF",
        ModelVariant.INTERN_VL3_5_14B_Q8_0: "OpenGVLab/InternVL3_5-14B-HF",
    }

    DEFAULT_VARIANT = ModelVariant.INTERN_VL3_5_4B_Q4_K_M

    sample_image = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="InternVL3.5 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        hf_config_name = self._HF_CONFIGS[self._variant]

        self.processor = AutoProcessor.from_pretrained(
            self._HF_PROCESSORS[self._variant],
            trust_remote_code=True,
        )

        # The GGUF file only contains the quantized text backbone (Qwen3) and
        # cannot be loaded via from_pretrained due to patching conflicts with
        # other loaders. Since this runs in compile-only mode, initialise from
        # config (random weights) — weight values are irrelevant for compilation.
        config = AutoConfig.from_pretrained(hf_config_name)

        if dtype_override is not None:
            config.torch_dtype = dtype_override
            config.text_config.torch_dtype = dtype_override

        model = InternVLForConditionalGeneration(config)

        if dtype_override is not None:
            model = model.to(dtype_override)

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
                    {"type": "text", "text": "What is shown in this image?"},
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
