# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bartowski xlangai Jedi-3B-1080p GGUF model loader implementation for image to text.
"""

import transformers.modeling_gguf_pytorch_utils as _gguf_utils

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
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


def _patch_gguf_qwen2_5_vl():
    """Patch GGUF weight-map lookup to support Qwen2.5-VL architecture.

    Qwen2_5_VLConfig stores num_hidden_layers inside text_config rather than
    at the top level, causing AttributeError in get_gguf_hf_weights_map.
    Additionally, the HF model_type "qwen2_5_vl" has no direct entry in
    gguf-py's MODEL_ARCH_NAMES, so we remap it to "qwen2vl".
    """
    _orig = _gguf_utils.get_gguf_hf_weights_map

    def _patched(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
        if num_layers is None and not hasattr(hf_model.config, "num_hidden_layers"):
            text_cfg = getattr(hf_model.config, "text_config", None)
            if text_cfg is not None:
                num_layers = getattr(text_cfg, "num_hidden_layers", None)
        if model_type is None:
            model_type = getattr(hf_model.config, "model_type", None)
        if model_type == "qwen2_5_vl":
            model_type = "qwen2vl"
        return _orig(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    _gguf_utils.get_gguf_hf_weights_map = _patched


_patch_gguf_qwen2_5_vl()


class ModelVariant(StrEnum):
    """Available Bartowski xlangai Jedi-3B-1080p GGUF variants for image to text."""

    XLANGAI_JEDI_3B_1080P_GGUF = "xlangai_jedi_3b_1080p_gguf"


class ModelLoader(ForgeModel):
    """Bartowski xlangai Jedi-3B-1080p GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.XLANGAI_JEDI_3B_1080P_GGUF: LLMModelConfig(
            pretrained_model_name="bartowski/xlangai_Jedi-3B-1080p-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XLANGAI_JEDI_3B_1080P_GGUF

    GGUF_FILE = "xlangai_Jedi-3B-1080p-Q4_K_M.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Bartowski xlangai Jedi-3B-1080p GGUF",
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

        # GGUF repos do not ship a processor; use the base model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
