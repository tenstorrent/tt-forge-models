# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 2.5 VL GGUF model loader implementation for vision-language tasks.
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Optional


def _patch_qwen2vl_gguf():
    """Monkey-patch transformers to support qwen2vl GGUF architecture."""
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "qwen2vl" in GGUF_SUPPORTED_ARCHITECTURES:
        return

    GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")

    GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen2vl"] = GGUF_TO_TRANSFORMERS_MAPPING[
        "config"
    ]["qwen2"].copy()

    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "qwen2vl":
            result["config"]["model_type"] = "qwen2_5_vl"
            rope_theta = result["config"].get("rope_theta", 1000000.0)
            result["config"]["rope_scaling"] = {
                "type": "mrope",
                "mrope_section": [16, 24, 24],
                "rope_theta": rope_theta,
                "rope_type": "default",
            }
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


_patch_qwen2vl_gguf()

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
from .src.model import Wrapper


class ModelVariant(StrEnum):
    """Available Qwen 2.5 VL GGUF model variants for vision-language tasks."""

    QWEN_2_5_VL_7B_INSTRUCT_GGUF = "7B_Instruct_GGUF"


# Map variants to their GGUF filenames
_GGUF_FILES = {
    ModelVariant.QWEN_2_5_VL_7B_INSTRUCT_GGUF: "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
}


class ModelLoader(ForgeModel):
    """Qwen 2.5 VL GGUF model loader implementation for vision-language tasks."""

    _VARIANTS = {
        ModelVariant.QWEN_2_5_VL_7B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN_2_5_VL_7B_INSTRUCT_GGUF

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
        return ModelInfo(
            model="Qwen 2.5-VL GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    # Processor is loaded from the original Qwen repo since the GGUF repo
    # only contains quantized model weights without tokenizer/processor configs.
    _PROCESSOR_SOURCE = "Qwen/Qwen2.5-VL-7B-Instruct"

    def _load_processor(self):
        processor_kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
        self.processor = AutoProcessor.from_pretrained(
            self._PROCESSOR_SOURCE, **processor_kwargs
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoConfig

        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = _GGUF_FILES[self._variant]

        config = AutoConfig.from_pretrained(self._PROCESSOR_SOURCE)
        config.use_cache = False

        model_kwargs = {"low_cpu_mem_usage": True, "use_cache": False, "config": config}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = gguf_file

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
