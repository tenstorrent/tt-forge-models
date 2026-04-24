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

import transformers.configuration_utils as _config_utils
import transformers.modeling_gguf_pytorch_utils as _gguf_utils
import transformers.models.auto.tokenization_auto as _auto_tokenizer
import transformers.tokenization_utils_tokenizers as _tok_utils
from transformers.modeling_gguf_pytorch_utils import (
    load_gguf_checkpoint as _orig_load_gguf_checkpoint,
    GGUF_SUPPORTED_ARCHITECTURES,
)


def _patch_qwen2vl_gguf_support():
    """Register qwen2vl as a supported GGUF architecture mapped to qwen2_vl.

    UI-TARS and similar models use GGUF files declaring architecture 'qwen2vl'
    which transformers 5.x does not recognise. The GGUF only stores the LM
    backbone; vision encoder weights are absent and will use default init,
    which is acceptable for compile-only tests.
    """
    if "qwen2vl" not in GGUF_SUPPORTED_ARCHITECTURES:
        GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")
    if "qwen2vl" not in _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]:
        _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen2vl"] = {
            "context_length": "max_position_embeddings",
            "block_count": "num_hidden_layers",
            "feed_forward_length": "intermediate_size",
            "embedding_length": "hidden_size",
            "rope.dimension_count": None,
            "rope.freq_base": "rope_theta",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "vocab_size": "vocab_size",
        }
    for section in ("tokenizer", "tokenizer_config"):
        mapping = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING.get(section, {})
        if "qwen2" in mapping:
            mapping.setdefault("qwen2vl", mapping["qwen2"])


def _read_gguf_mrope_section(gguf_path):
    """Read rope.dimension_sections from a GGUF file and return as a list."""
    try:
        from gguf import GGUFReader

        reader = GGUFReader(gguf_path)
        field_name = "qwen2vl.rope.dimension_sections"
        if field_name not in reader.fields:
            return None
        field = reader.fields[field_name]
        # field.data indexes into field.parts; parts are int32 values
        values = [
            int(field.parts[i][0]) for i in field.data if int(field.parts[i][0]) != 0
        ]
        return values if values else None
    except Exception:
        return None


def _patched_load_gguf_checkpoint(*args, **kwargs):
    _patch_qwen2vl_gguf_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    cfg = result.get("config", {})
    if cfg.get("model_type") == "qwen2vl":
        cfg["model_type"] = "qwen2_vl"
        # The GGUF loader doesn't extract rope.dimension_sections into
        # rope_parameters, so we read it directly and inject it.
        gguf_path = args[0] if args else kwargs.get("gguf_checkpoint_path")
        if gguf_path:
            mrope_section = _read_gguf_mrope_section(gguf_path)
            if mrope_section:
                cfg.setdefault("rope_parameters", {})["mrope_section"] = mrope_section
    return result


_orig_get_gguf_hf_weights_map = _gguf_utils.get_gguf_hf_weights_map


def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, qual_name=""
):
    if model_type is None and hf_model is not None:
        model_type = hf_model.config.model_type
    if model_type == "qwen2_vl":
        if num_layers is None and hf_model is not None:
            num_layers = hf_model.config.text_config.num_hidden_layers
        model_type = "qwen2vl"
    return _orig_get_gguf_hf_weights_map(
        hf_model, processor, model_type, num_layers, qual_name
    )


_patch_qwen2vl_gguf_support()
_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_config_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_auto_tokenizer.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_tok_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint
_gguf_utils.get_gguf_hf_weights_map = _patched_get_gguf_hf_weights_map


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
            pretrained_model_name, **model_kwargs
        )
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
