# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UI-TARS 1.5 7B mradermacher GGUF model loader implementation for vision-language GUI agent tasks.

Repository:
- https://huggingface.co/mradermacher/UI-TARS-1.5-7B-GGUF
"""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Optional


from ...base import ForgeModel


def _patch_transformers_qwen2vl_gguf():
    """Monkey-patch transformers to add qwen2vl GGUF architecture support.

    Transformers 5.x has Qwen2VLForConditionalGeneration but lacks GGUF loading
    support for the qwen2vl architecture. We register the architecture and inject
    a text_config sub-dict that Qwen2VLConfig requires.
    """
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
    )
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )

    if "qwen2vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen2vl"] = {
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

    if "qwen2vl" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["qwen2vl"] = GGUFQwen2Converter

    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "qwen2vl":
            text_param_keys = [
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "rms_norm_eps",
                "max_position_embeddings",
                "rope_theta",
                "vocab_size",
                "tie_word_embeddings",
            ]
            text_config = {k: config[k] for k in text_param_keys if k in config}
            # Qwen2-VL requires mrope_section in rope_parameters for multimodal RoPE.
            # All Qwen2-VL variants use head_dim=128 → mrope_section=[16,24,24].
            text_config["rope_parameters"] = {
                "rope_theta": text_config.pop("rope_theta", 1000000.0),
                "rope_type": "mrope",
                "mrope_section": [16, 24, 24],
            }
            config["text_config"] = text_config
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint
    for mod in (config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_transformers_qwen2vl_gguf()
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
    """Available UI-TARS 1.5 7B mradermacher GGUF model variants."""

    Q4_K_M = "Q4_K_M"
    Q8_0 = "Q8_0"


class ModelLoader(ForgeModel):
    """UI-TARS 1.5 7B mradermacher GGUF model loader implementation for vision-language GUI agent tasks."""

    _VARIANTS = {
        ModelVariant.Q4_K_M: ModelConfig(
            pretrained_model_name="mradermacher/UI-TARS-1.5-7B-GGUF",
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name="mradermacher/UI-TARS-1.5-7B-GGUF",
        ),
    }

    _GGUF_FILES = {
        ModelVariant.Q4_K_M: "UI-TARS-1.5-7B.Q4_K_M.gguf",
        ModelVariant.Q8_0: "UI-TARS-1.5-7B.Q8_0.gguf",
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
            model="UI-TARS 1.5 7B mradermacher GGUF",
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
