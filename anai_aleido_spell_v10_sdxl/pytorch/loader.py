# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ANAI Aleido Spell v10 SDXL (John6666/anai-aleido-spell-v10-sdxl) model loader
implementation.

ANAI Aleido Spell v10 SDXL is an anime-focused Stable Diffusion XL merge
model (NoobAI-XL + Illustrious-XL) for text-to-image generation.

Available variants:
- ANAI_ALEIDO_SPELL_V10_SDXL: John6666/anai-aleido-spell-v10-sdxl text-to-image generation
"""

from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline

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


REPO_ID = "John6666/anai-aleido-spell-v10-sdxl"


class ModelVariant(StrEnum):
    """Available ANAI Aleido Spell v10 SDXL model variants."""

    ANAI_ALEIDO_SPELL_V10_SDXL = "anai-aleido-spell-v10-sdxl"


class ModelLoader(ForgeModel):
    """ANAI Aleido Spell v10 SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.ANAI_ALEIDO_SPELL_V10_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.ANAI_ALEIDO_SPELL_V10_SDXL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="anai-aleido-spell-v10-sdxl",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ANAI Aleido Spell v10 SDXL pipeline.

        Returns:
            StableDiffusionXLPipeline: The ANAI Aleido Spell v10 SDXL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            use_safetensors=True,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the ANAI Aleido Spell v10 SDXL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
