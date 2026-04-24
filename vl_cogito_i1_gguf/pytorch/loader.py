# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mradermacher/VL-Cogito-i1-GGUF model loader for vision-language tasks.
"""
import importlib
import importlib.metadata
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import Optional

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
    """Available mradermacher/VL-Cogito-i1-GGUF model variants."""

    VL_COGITO_I1_GGUF = "i1_GGUF"


class ModelLoader(ForgeModel):
    """mradermacher/VL-Cogito-i1-GGUF model loader for vision-language tasks."""

    _VARIANTS = {
        ModelVariant.VL_COGITO_I1_GGUF: LLMModelConfig(
            pretrained_model_name="mradermacher/VL-Cogito-i1-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VL_COGITO_I1_GGUF

    GGUF_FILE = "VL-Cogito.i1-Q4_K_M.gguf"

    # Processor is loaded from the base Qwen2.5-VL-7B-Instruct repo since the
    # GGUF repo only contains quantized model weights without tokenizer/processor
    # configs. VL-Cogito is fine-tuned from Qwen/Qwen2.5-VL-7B-Instruct.
    PROCESSOR_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

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

    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="mradermacher VL-Cogito i1 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _refresh_gguf_detection():
        import inspect
        import transformers.utils.import_utils as _tx
        import transformers.modeling_gguf_pytorch_utils as _gguf_utils

        _tx.PACKAGE_DISTRIBUTION_MAPPING = importlib.metadata.packages_distributions()
        _tx.is_gguf_available.cache_clear()

        try:
            sig = inspect.signature(_gguf_utils.load_gguf_checkpoint)
            has_model_to_load = "model_to_load" in sig.parameters
        except (ValueError, TypeError):
            has_model_to_load = False

        if not has_model_to_load:
            importlib.reload(_gguf_utils)

        # transformers 5.x doesn't include qwen2vl in GGUF_SUPPORTED_ARCHITECTURES;
        # the GGUF contains only LLM backbone — vision encoder stays randomly initialized.
        if "qwen2vl" not in _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
            _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"][
                "qwen2vl"
            ] = _gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen2"].copy()
            _gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen2vl")

    def _load_processor(self):
        processor_kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
        self.processor = AutoProcessor.from_pretrained(
            self.PROCESSOR_MODEL, **processor_kwargs
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        self._refresh_gguf_detection()

        from transformers import AutoConfig

        # GGUF repo has no config.json; load full vision+language config from base model
        config = AutoConfig.from_pretrained(self.PROCESSOR_MODEL)

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True, "use_cache": False, "config": config}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

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
