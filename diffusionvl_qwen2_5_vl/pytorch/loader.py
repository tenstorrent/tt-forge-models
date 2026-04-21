# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DiffusionVL Qwen 2.5 VL model loader implementation for vision-language tasks.

hustvl/DiffusionVL-Qwen2.5VL-3B translates the autoregressive Qwen2.5-VL-3B
model into a diffusion vision language model using a block decoding strategy.
It ships custom modeling code via ``trust_remote_code`` and is exposed through
``AutoModelForCausalLM``.
"""
import requests
import torch
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
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
    """Available DiffusionVL Qwen 2.5 VL model variants."""

    DIFFUSIONVL_QWEN_2_5_VL_3B = "3B"


class ModelLoader(ForgeModel):
    """DiffusionVL Qwen 2.5 VL model loader implementation for vision-language tasks."""

    _VARIANTS = {
        ModelVariant.DIFFUSIONVL_QWEN_2_5_VL_3B: LLMModelConfig(
            pretrained_model_name="hustvl/DiffusionVL-Qwen2.5VL-3B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DIFFUSIONVL_QWEN_2_5_VL_3B

    sample_image_url = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DiffusionVL-Qwen2.5-VL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DiffusionVL Qwen 2.5 VL model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Wrapped DiffusionVL model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "low_cpu_mem_usage": True,
            "use_cache": False,
            "trust_remote_code": True,
        }

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the DiffusionVL Qwen 2.5 VL model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
                           If specified, converts pixel_values to the specified dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        image = Image.open(
            BytesIO(requests.get(self.sample_image_url).content)
        ).convert("RGB")

        text = self.processor.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
