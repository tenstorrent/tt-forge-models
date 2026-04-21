# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Anime Desire Illustrious (AI-Porn/anime-desire-illustrious) model loader implementation.

Anime Desire Illustrious is an anime-focused Stable Diffusion XL checkpoint
derived from the Illustrious-XL family, distributed as a single fp16 safetensors
file for text-to-image generation.

Available variants:
- ANIME_DESIRE_ILLUSTRIOUS_V3: AI-Porn/anime-desire-illustrious text-to-image generation
"""

from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download

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

REPO_ID = "AI-Porn/anime-desire-illustrious"
CHECKPOINT_FILE = "anime-2d-ill_v3.fp16.safetensors"


class ModelVariant(StrEnum):
    """Available Anime Desire Illustrious model variants."""

    ANIME_DESIRE_ILLUSTRIOUS_V3 = "anime-desire-illustrious-v3"


class ModelLoader(ForgeModel):
    """Anime Desire Illustrious model loader implementation."""

    _VARIANTS = {
        ModelVariant.ANIME_DESIRE_ILLUSTRIOUS_V3: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.ANIME_DESIRE_ILLUSTRIOUS_V3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Anime Desire Illustrious",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Anime Desire Illustrious pipeline from single-file checkpoint.

        Returns:
            StableDiffusionXLPipeline: The loaded pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        model_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=CHECKPOINT_FILE,
        )
        self.pipeline = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
