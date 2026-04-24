# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit-Rapid GGUF model loader implementation.

Loads GGUF-quantized Qwen2VL vision-language model variants from
Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF. Built on the Qwen2.5-VL-7B-Instruct
architecture, fine-tuned for rapid image editing.

Available variants:
- V9_0_Q4_K_M: v9.0 Q4_K_M quantization (13.3 GB)
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig
from typing import Optional

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS


def _patch_qwen_image_support():
    """Register qwen_image architecture as an alias for qwen2.

    The Qwen-Image-Edit-Rapid GGUF files declare architecture as 'qwen_image'
    which transformers does not recognise. The underlying model is Qwen2.5-VL
    (qwen2 family).
    """
    if "qwen_image" in GGUF_SUPPORTED_ARCHITECTURES:
        return
    GGUF_SUPPORTED_ARCHITECTURES.append("qwen_image")
    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen2" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section][
                "qwen_image"
            ] = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen2"]
    if "qwen2" in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen_image"] = GGUF_TO_FAST_CONVERTERS["qwen2"]


def _patched_load_gguf_checkpoint(*args, **kwargs):
    _patch_qwen_image_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    if result.get("config", {}).get("model_type") == "qwen_image":
        result["config"]["model_type"] = "qwen2"
    return result


_patch_qwen_image_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

REPO_ID = "Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF"

_GGUF_FILES = {
    "v9.0_Q4_K_M": "v90/qwen-rapid-nsfw-v9.0-Q4_K_M.gguf",
}

# Vision processing parameters
MIN_PIXELS = 56 * 56
MAX_PIXELS = 13 * 28 * 1280


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit-Rapid GGUF model variants."""

    V9_0_Q4_K_M = "v9.0_Q4_K_M"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit-Rapid GGUF model loader for vision-language image editing."""

    _VARIANTS = {
        ModelVariant.V9_0_Q4_K_M: LLMModelConfig(
            pretrained_model_name=REPO_ID,
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V9_0_Q4_K_M

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
            model="QWEN_IMAGE_EDIT_RAPID_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_gguf_file(self) -> str:
        """Get the GGUF filename for the current variant."""
        return _GGUF_FILES[self._variant.value]

    def _load_processor(self):
        """Load the vision-language processor."""
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )
        return self.processor

    _BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGUF-quantized Qwen2VL model.

        Returns:
            Qwen2_5_VLForConditionalGeneration instance.
        """
        if self.processor is None:
            self._load_processor()

        gguf_file = self._get_gguf_file()

        config = AutoConfig.from_pretrained(self._BASE_MODEL)
        if self.num_layers is not None:
            config.num_hidden_layers = self.num_layers

        model_kwargs = {"config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = gguf_file

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the Qwen2VL model.

        Returns:
            dict: Input tensors for the model's forward pass.
        """
        if self.processor is None:
            self._load_processor()

        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(self.messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(self._BASE_MODEL)
        return self.config
