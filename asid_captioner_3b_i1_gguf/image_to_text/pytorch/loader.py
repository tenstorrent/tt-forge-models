# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ASID Captioner 3B i1 GGUF model loader implementation for image to text.
"""

from transformers import (
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
    """Available ASID Captioner 3B i1 GGUF model variants for image to text."""

    ASID_CAPTIONER_3B_I1_Q4_K_M_GGUF = "3b_i1_Q4_K_M_gguf"


class ModelLoader(ForgeModel):
    """ASID Captioner 3B i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.ASID_CAPTIONER_3B_I1_Q4_K_M_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/ASID-Captioner-3B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ASID_CAPTIONER_3B_I1_Q4_K_M_GGUF

    GGUF_FILE = "ASID-Captioner-3B.i1-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ASID Captioner 3B i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        # gguf lacks __version__; set it so transformers' is_gguf_available() can parse it
        try:
            import importlib.metadata

            import gguf as _gguf

            if not hasattr(_gguf, "__version__"):
                _gguf.__version__ = importlib.metadata.version("gguf")
        except Exception:
            pass

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained(
            "AudioVisual-Caption/ASID-Captioner-3B"
        )

        model = Qwen2VLForConditionalGeneration.from_pretrained(
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
