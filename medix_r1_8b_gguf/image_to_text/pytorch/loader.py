# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MediX R1 8B GGUF model loader implementation for image to text.
"""

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
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
    """Available MediX R1 8B GGUF model variants for image to text."""

    MEDIX_R1_8B_Q4_K_M = "8b_q4_k_m"


class ModelLoader(ForgeModel):
    """MediX R1 8B GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.MEDIX_R1_8B_Q4_K_M: LLMModelConfig(
            pretrained_model_name="MBZUAI/MediX-R1-8B-GGUF",
            max_length=128,
        ),
    }

    _GGUF_FILES = {
        ModelVariant.MEDIX_R1_8B_Q4_K_M: "MediX-R1-8B-Q4_K_M.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.MEDIX_R1_8B_Q4_K_M

    BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

    sample_image = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @staticmethod
    def _fix_gguf_qwen3vl():
        """Register qwen3vl GGUF architecture so transformers can load it.

        The GGUF file stores architecture as "qwen3vl" which transformers does
        not yet support. We alias it to qwen3's config mapping (text backbone is
        identical) and patch load_gguf_checkpoint to set the correct model_type
        for Qwen3VLForConditionalGeneration.
        """
        import transformers.configuration_utils as _config_utils
        import transformers.modeling_gguf_pytorch_utils as _gguf_utils
        import transformers.modeling_utils as _modeling_utils

        if "qwen3vl" in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
            return

        _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"][
            "qwen3vl"
        ] = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3"]
        _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

        _orig_load = _gguf_utils.load_gguf_checkpoint

        def _patched_load(*args, **kwargs):
            result = _orig_load(*args, **kwargs)
            if isinstance(result, dict) and isinstance(result.get("config"), dict):
                if result["config"].get("model_type") == "qwen3vl":
                    result["config"]["model_type"] = "qwen3_vl"
                    result["config"]["architectures"] = [
                        "Qwen3VLForConditionalGeneration"
                    ]
            return result

        _gguf_utils.load_gguf_checkpoint = _patched_load
        _config_utils.load_gguf_checkpoint = _patched_load
        _modeling_utils.load_gguf_checkpoint = _patched_load

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MediX R1 8B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self._fix_gguf_qwen3vl()

        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs["gguf_file"] = gguf_file
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(self.BASE_MODEL)

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
