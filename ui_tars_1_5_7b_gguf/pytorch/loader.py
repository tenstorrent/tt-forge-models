# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UI-TARS 1.5 7B GGUF model loader implementation for vision-language GUI agent tasks.

Repository:
- https://huggingface.co/Mungert/UI-TARS-1.5-7B-GGUF
"""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Optional

import transformers.modeling_gguf_pytorch_utils as _gguf_utils
from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFQwen2Converter

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model import Wrapper


def _patch_qwen2vl_gguf_support():
    """Register qwen2vl architecture in transformers GGUF loading tables.

    transformers does not yet include qwen2vl in its GGUF conversion tables.
    We add it by copying the qwen2 config mapping, which shares the same text
    config structure as the VL variant. We also patch get_gguf_hf_weights_map
    to handle Qwen2VLConfig's nested text_config.num_hidden_layers.
    """
    if "qwen2vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    for section in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING:
        if "qwen2" in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen2vl"] = dict(
                _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING[section]["qwen2"]
            )

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")

    if "qwen2vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen2vl"] = GGUFQwen2Converter

    _orig_get_gguf_hf_weights_map = _gguf_utils.get_gguf_hf_weights_map

    def _patched_get_gguf_hf_weights_map(
        hf_model, processor, model_type=None, num_layers=None, qual_name=""
    ):
        if model_type is None:
            model_type = hf_model.config.model_type
        if model_type in ("qwen2vl", "qwen2_vl"):
            model_type = "qwen2vl"
            if num_layers is None:
                num_layers = hf_model.config.text_config.num_hidden_layers
        return _orig_get_gguf_hf_weights_map(
            hf_model, processor, model_type, num_layers, qual_name
        )

    _gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


_patch_qwen2vl_gguf_support()


class ModelVariant(StrEnum):
    """Available UI-TARS 1.5 7B GGUF model variants."""

    Q4_K = "Q4_K"
    Q8_0 = "Q8_0"


class ModelLoader(ForgeModel):
    """UI-TARS 1.5 7B GGUF model loader implementation for vision-language GUI agent tasks."""

    _VARIANTS = {
        ModelVariant.Q4_K: ModelConfig(
            pretrained_model_name="Mungert/UI-TARS-1.5-7B-GGUF",
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name="Mungert/UI-TARS-1.5-7B-GGUF",
        ),
    }

    _GGUF_FILES = {
        ModelVariant.Q4_K: "UI-TARS-1.5-7B-q4_k.gguf",
        ModelVariant.Q8_0: "UI-TARS-1.5-7B-q8_0.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.Q8_0

    # Processor source (original non-GGUF model)
    _PROCESSOR_MODEL = "ByteDance-Seed/UI-TARS-1.5-7B"

    # Shared configuration parameters
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

    # Vision processing parameters
    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="UI-TARS 1.5 7B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor from the original non-GGUF model."""
        processor_kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }

        self.processor = AutoProcessor.from_pretrained(
            self._PROCESSOR_MODEL, **processor_kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        model_kwargs = {"low_cpu_mem_usage": True}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = gguf_file

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, ignore_mismatched_sizes=True, **model_kwargs
        )

        # GGUF loading doesn't populate mrope_section from qwen2vl.rope.dimension_sections
        if "mrope_section" not in model.config.text_config.rope_parameters:
            model.config.text_config.rope_parameters = {
                **model.config.text_config.rope_parameters,
                "mrope_section": [16, 24, 24],
            }

        model.eval()
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
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

        return inputs
