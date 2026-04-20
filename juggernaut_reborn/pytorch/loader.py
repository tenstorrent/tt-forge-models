# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Juggernaut Reborn (HyperX-Sentience/Juggernaut-Reborn) model loader implementation.

Juggernaut Reborn is a photorealistic text-to-image model based on Stable Diffusion
v1.5, distributed as a single-file safetensors checkpoint.

Available variants:
- BASE: HyperX-Sentience/Juggernaut-Reborn text-to-image generation
"""

from typing import Optional

import torch
from diffusers import StableDiffusionPipeline
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

REPO_ID = "HyperX-Sentience/Juggernaut-Reborn"
CHECKPOINT_FILE = "juggernaut_reborn.safetensors"


class ModelVariant(StrEnum):
    """Available Juggernaut Reborn model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Juggernaut Reborn model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Juggernaut Reborn",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Juggernaut Reborn pipeline from a single-file checkpoint.

        Returns:
            StableDiffusionPipeline: The Juggernaut Reborn pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        model_path = hf_hub_download(repo_id=REPO_ID, filename=CHECKPOINT_FILE)
        self.pipeline = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Juggernaut Reborn model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "a photo of an astronaut riding a horse on mars",
        ] * batch_size
