# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
unsloth/Qwen2.5-VL-3B-Instruct-GGUF model loader for vision-language tasks.
"""
import os
import torch
from huggingface_hub import try_to_load_from_cache
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig
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
    """Available unsloth/Qwen2.5-VL-3B-Instruct-GGUF model variants."""

    UNSLOTH_QWEN2_5_VL_3B_INSTRUCT_GGUF = "3B_Instruct_GGUF"


class ModelLoader(ForgeModel):
    """unsloth/Qwen2.5-VL-3B-Instruct-GGUF model loader for vision-language tasks."""

    _VARIANTS = {
        ModelVariant.UNSLOTH_QWEN2_5_VL_3B_INSTRUCT_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen2.5-VL-3B-Instruct-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNSLOTH_QWEN2_5_VL_3B_INSTRUCT_GGUF

    GGUF_FILE = "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"

    # Processor is loaded from the original Qwen repo since the GGUF repo
    # only contains quantized model weights without tokenizer/processor configs.
    PROCESSOR_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

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
            model="unsloth Qwen2.5-VL-3B-Instruct GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

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
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        # unsloth GGUF repo has no config.json in the HF repo root so
        # from_pretrained(repo_id) fails. Use try_to_load_from_cache to
        # find the locally cached GGUF, then point from_pretrained at the
        # snapshot directory which has both config.json and the GGUF file.
        cached = try_to_load_from_cache(
            repo_id=pretrained_model_name, filename=self.GGUF_FILE
        )
        if cached is not None:
            model_source = os.path.dirname(cached)
        else:
            model_source = pretrained_model_name
        model_kwargs["gguf_file"] = self.GGUF_FILE

        # When gguf_file is specified, transformers loads the config from GGUF
        # metadata rather than config.json. The GGUF for qwen2vl only contains
        # text fields, so the vision config would be empty and cause dimension
        # mismatches. Pre-load config from config.json and pass it explicitly
        # so the full vision config is used while GGUF provides the weights.
        config = AutoConfig.from_pretrained(model_source)
        # get_gguf_hf_weights_map expects num_hidden_layers and model_type at
        # the top level. Qwen2_5_VLConfig is composite; add these aliases so
        # GGUF tensor name mapping resolves against the "qwen2vl" arch.
        config.num_hidden_layers = config.text_config.num_hidden_layers
        config.model_type = "qwen2vl"

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_source, config=config, **model_kwargs
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
