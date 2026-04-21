# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EarthnDusk Loras_2023 Stable Diffusion model loader implementation.

Loads the Stable Diffusion v1.5 base pipeline and applies a LoRA weight file
from EarthnDusk/Loras_2023, a backup collection of SD1.5 LoRA adapters.
"""

import torch
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from diffusers import StableDiffusionPipeline
from typing import Optional


class ModelVariant(StrEnum):
    """Available EarthnDusk Loras_2023 model variants."""

    DUSK_ART_V7 = "dusk_art_v7"


class ModelLoader(ForgeModel):
    """EarthnDusk Loras_2023 Stable Diffusion model loader implementation."""

    _VARIANTS = {
        ModelVariant.DUSK_ART_V7: ModelConfig(
            pretrained_model_name="EarthnDusk/Loras_2023",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DUSK_ART_V7

    _LORA_WEIGHT_NAMES = {
        ModelVariant.DUSK_ART_V7: "Jan_11_SD15/DuskArtV7.safetensors",
    }

    _BASE_MODEL = "runwayml/stable-diffusion-v1-5"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="EarthnDusk Loras_2023",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override or torch.bfloat16
        pipe = StableDiffusionPipeline.from_pretrained(
            self._BASE_MODEL, torch_dtype=dtype, **kwargs
        )
        pipe.load_lora_weights(
            self._variant_config.pretrained_model_name,
            weight_name=self._LORA_WEIGHT_NAMES[self._variant],
        )
        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1):
        prompt = [
            "a mystical fantasy landscape, dusk_art style, vibrant colors, detailed",
        ] * batch_size
        return prompt
