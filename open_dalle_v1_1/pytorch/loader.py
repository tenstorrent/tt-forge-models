# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenDalle V1.1 (dataautogpt3/OpenDalleV1.1) model loader implementation.

OpenDalle V1.1 is a text-to-image model based on the Stable Diffusion XL
architecture, produced via a proprietary merging method and additional
tuning on top of SDXL.

Available variants:
- OPEN_DALLE_V1_1: dataautogpt3/OpenDalleV1.1 text-to-image generation
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


REPO_ID = "dataautogpt3/OpenDalleV1.1"


class ModelVariant(StrEnum):
    """Available OpenDalle V1.1 model variants."""

    OPEN_DALLE_V1_1 = "OpenDalleV1.1"


class ModelLoader(ForgeModel):
    """OpenDalle V1.1 model loader implementation."""

    _VARIANTS = {
        ModelVariant.OPEN_DALLE_V1_1: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.OPEN_DALLE_V1_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="OpenDalleV1.1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the OpenDalle V1.1 pipeline.

        Returns:
            StableDiffusionXLPipeline: The OpenDalle V1.1 pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the OpenDalle V1.1 model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "black fluffy gorgeous dangerous cat animal creature, large orange eyes, "
            "big fluffy ears, piercing gaze, full moon, dark ambiance, best quality, "
            "extremely detailed"
        ] * batch_size
