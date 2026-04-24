# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GUI-Libra 4B i1 GGUF model loader implementation for image to text.
"""

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.configuration_utils as _config_utils
from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
from transformers.integrations.ggml import GGUF_CONFIG_MAPPING

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoConfig,
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


def _patch_qwen3vl_gguf_support():
    """Register qwen3vl GGUF architecture for transformers.

    Qwen3-VL GGUF files only contain text model weights (language model backbone).
    The vision encoder is not quantized and keeps its default initialization.
    The text model parameters use the same GGUF naming convention as qwen3.
    """
    if "qwen3vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    # Register config field mapping (same as qwen3 text backbone)
    GGUF_CONFIG_MAPPING["qwen3vl"] = dict(GGUF_CONFIG_MAPPING["qwen3"])
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen3vl")

    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen3" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section][
                "qwen3vl"
            ] = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen3"]

    # Patch get_gguf_hf_weights_map to translate qwen3_vl (HF) → qwen3vl (gguf-py)
    _orig_get_weights_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = getattr(getattr(hf_model, "config", None), "model_type", None)
        if model_type == "qwen3_vl":
            model_type = "qwen3vl"
        return _orig_get_weights_map(
            hf_model,
            processor,
            model_type=model_type,
            num_layers=num_layers,
            qual_name=qual_name,
        )

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_weights_map


_patch_qwen3vl_gguf_support()


class ModelVariant(StrEnum):
    """Available GUI-Libra 4B i1 GGUF model variants for image to text."""

    GUI_LIBRA_4B_I1_GGUF = "4B_i1_GGUF"


class ModelLoader(ForgeModel):
    """GUI-Libra 4B i1 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.GUI_LIBRA_4B_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/GUI-Libra-4B-i1-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GUI_LIBRA_4B_I1_GGUF

    GGUF_FILE = "GUI-Libra-4B.i1-Q4_K_M.gguf"

    # Base model provides the full config (text + vision) and processor
    BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GUI-Libra 4B i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Pre-load config from base model to include the correct vision config.
        # The GGUF file only contains text model weights; vision encoder defaults
        # come from the 4B base model config.
        config = AutoConfig.from_pretrained(self.BASE_MODEL)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs["config"] = config
        model_kwargs |= kwargs

        # GGUF repos do not ship a processor; use the base model
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
