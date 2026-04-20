# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InternVL3.5 GGUF model loader implementation for image to text.

The GGUF checkpoint only contains the text backbone (Qwen3 architecture).
We load the vision config from the base (non-GGUF) model and combine
them into a full InternVLConfig, following the same pattern as GLM-OCR GGUF.
"""
import importlib.metadata

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoConfig,
    InternVLConfig,
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


def _refresh_gguf_detection():
    """Refresh transformers' gguf package detection if the package was installed after import."""
    from transformers.utils import import_utils

    if "gguf" not in import_utils.PACKAGE_DISTRIBUTION_MAPPING:
        import_utils.PACKAGE_DISTRIBUTION_MAPPING = (
            importlib.metadata.packages_distributions()
        )
        import_utils.is_gguf_available.cache_clear()


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

    _GGUF_FILES = {
        ModelVariant.INTERN_VL3_5_4B_Q4_K_M: "OpenGVLab_InternVL3_5-4B-Q4_K_M.gguf",
        ModelVariant.INTERN_VL3_5_4B_Q8_0: "OpenGVLab_InternVL3_5-4B-Q8_0.gguf",
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: "OpenGVLab_InternVL3_5-14B-Q4_K_M.gguf",
        ModelVariant.INTERN_VL3_5_14B_Q8_0: "OpenGVLab_InternVL3_5-14B-Q8_0.gguf",
    }

    _HF_PROCESSORS = {
        ModelVariant.INTERN_VL3_5_4B_Q4_K_M: "OpenGVLab/InternVL3_5-4B-HF",
        ModelVariant.INTERN_VL3_5_4B_Q8_0: "OpenGVLab/InternVL3_5-4B-HF",
        ModelVariant.INTERN_VL3_5_14B_Q4_K_M: "OpenGVLab/InternVL3_5-14B-HF",
        ModelVariant.INTERN_VL3_5_14B_Q8_0: "OpenGVLab/InternVL3_5-14B-HF",
    }

    DEFAULT_VARIANT = ModelVariant.INTERN_VL3_5_4B_Q4_K_M

    _BASE_MODEL = "OpenGVLab/InternVL3_5-14B-HF"

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

    def _build_full_config(self):
        """Build a full InternVLConfig by wrapping the GGUF text config with a vision config."""
        _refresh_gguf_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]
        text_config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=gguf_file
        )
        base_config = AutoConfig.from_pretrained(self._BASE_MODEL)
        base_dict = base_config.to_dict()
        extra_keys = {
            k: v
            for k, v in base_dict.items()
            if k
            not in (
                "text_config",
                "vision_config",
                "transformers_version",
                "architectures",
                "model_type",
                "_name_or_path",
            )
        }
        return InternVLConfig(
            text_config=text_config.to_dict(),
            vision_config=base_config.vision_config.to_dict(),
            **extra_keys,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _refresh_gguf_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["gguf_file"] = gguf_file
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(
            self._HF_PROCESSORS[self._variant],
            trust_remote_code=True,
        )

        config = self._build_full_config()
        model_kwargs["config"] = config

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
