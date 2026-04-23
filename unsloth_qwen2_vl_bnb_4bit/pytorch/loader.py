# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unsloth Qwen2 VL BnB 4-bit model loader implementation for vision-language tasks.
"""
from typing import Optional

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from ...base import ForgeModel
from ...config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from .src.model import Wrapper


class ModelVariant(StrEnum):
    """Available Unsloth Qwen2 VL BnB 4-bit model variants for vision-language tasks."""

    UNSLOTH_QWEN_2_VL_7B_INSTRUCT_BNB_4BIT = "unsloth_7B_Instruct_bnb_4bit"


class ModelLoader(ForgeModel):
    """Unsloth Qwen2 VL BnB 4-bit model loader for vision-language tasks."""

    _VARIANTS = {
        ModelVariant.UNSLOTH_QWEN_2_VL_7B_INSTRUCT_BNB_4BIT: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNSLOTH_QWEN_2_VL_7B_INSTRUCT_BNB_4BIT

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
            model="Unsloth Qwen2 VL BnB 4-bit",
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
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self.processor

    @staticmethod
    def _replace_linear4bit(model, dtype):
        """Replace Linear4bit layers with regular Linear layers for CPU inference.

        BnB 4-bit quantization cannot run on CPU: the quant state isn't initialized
        when device_map="cpu", causing an assertion in fix_4bit_weight_quant_state_from_module.
        The weights are already loaded as float by transformers, so we just rewrap them.
        """
        try:
            import bitsandbytes as bnb
            import bitsandbytes.functional as bnbF
        except ImportError:
            return model

        for name, module in list(model.named_children()):
            if isinstance(module, bnb.nn.Linear4bit):
                weight = module.weight.data
                if weight.dtype == torch.uint8:
                    quant_state = getattr(
                        module.weight, "quant_state", None
                    ) or getattr(module, "quant_state", None)
                    if quant_state is not None:
                        weight = bnbF.dequantize_4bit(weight, quant_state).to(dtype)
                    else:
                        weight = weight.to(dtype)
                else:
                    weight = weight.to(dtype)

                new_linear = torch.nn.Linear(
                    weight.shape[1],
                    weight.shape[0],
                    bias=module.bias is not None,
                    dtype=dtype,
                )
                new_linear.weight = torch.nn.Parameter(weight)
                if module.bias is not None:
                    new_linear.bias = torch.nn.Parameter(module.bias.data.to(dtype))
                setattr(model, name, new_linear)
            else:
                ModelLoader._replace_linear4bit(module, dtype)

        return model

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Unsloth Qwen2 VL BnB 4-bit model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The wrapped Qwen2 VL BnB 4-bit model for vision-language tasks.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.float32

        model_kwargs = {"low_cpu_mem_usage": True, "device_map": "cpu"}
        model_kwargs["torch_dtype"] = dtype
        model_kwargs |= kwargs

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        ModelLoader._replace_linear4bit(model, dtype)
        model.eval()
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Unsloth Qwen2 VL BnB 4-bit model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
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

        return inputs
