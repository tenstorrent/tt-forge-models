# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MLbackup Lora_scrape_2023 LoRA Stable Diffusion model loader implementation
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
    """Available MLbackup Lora_scrape_2023 model variants."""

    DARK_FANTASY_V2 = "Dark_FantasyV2"


class ModelLoader(ForgeModel):
    """MLbackup Lora_scrape_2023 LoRA Stable Diffusion model loader implementation."""

    _VARIANTS = {
        ModelVariant.DARK_FANTASY_V2: ModelConfig(
            pretrained_model_name="MLbackup/Lora_scrape_2023",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DARK_FANTASY_V2

    _LORA_WEIGHT_NAMES = {
        ModelVariant.DARK_FANTASY_V2: "Dark_FantasyV2.safetensors",
    }

    _BASE_MODEL = "runwayml/stable-diffusion-v1-5"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="MLbackup Lora_scrape_2023",
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
            "a beautiful fantasy illustration, detailed artwork, masterpiece",
        ] * batch_size
        return prompt
