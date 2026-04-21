# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen Image Edit 2509 White Film to Rendering LoRA model loader implementation.

Loads the dx8152/Qwen-Image-Edit-2509-White_film_to_rendering LoRA adapter on
top of the Qwen/Qwen-Image-Edit-2509 base diffusion pipeline for converting
white-model (white film) renders into textured/material renderings.
"""

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available Qwen Image Edit 2509 White Film to Rendering model variants."""

    DEFAULT = "default"


class ModelLoader(ForgeModel):
    """Qwen Image Edit 2509 White Film to Rendering LoRA model loader."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="dx8152/Qwen-Image-Edit-2509-White_film_to_rendering",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    _BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"

    _LORA_WEIGHT_NAMES = {
        ModelVariant.DEFAULT: "白膜转材质.safetensors",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="Qwen Image Edit 2509 White Film to Rendering",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override or torch.bfloat16
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            self._BASE_MODEL, torch_dtype=dtype, **kwargs
        )
        pipe.load_lora_weights(
            self._variant_config.pretrained_model_name,
            weight_name=self._LORA_WEIGHT_NAMES[self._variant],
        )
        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1):
        image = Image.new("RGB", (512, 512), color=(255, 255, 255))
        prompt = "白膜转材质"
        return {
            "image": image,
            "prompt": prompt,
            "negative_prompt": " ",
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "true_cfg_scale": 4.0,
        }
