# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
One Obsession v16 NoobAI SDXL (John6666/one-obsession-v16-noobai-sdxl)
model loader implementation.

One Obsession v16 is an SDXL-based anime finetune merged from
Illustrious-XL-v2.0 and noobai-XL-1.0 for text-to-image generation.

Available variants:
- ONE_OBSESSION_V16_NOOBAI_SDXL: John6666/one-obsession-v16-noobai-sdxl text-to-image generation
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


REPO_ID = "John6666/one-obsession-v16-noobai-sdxl"


class ModelVariant(StrEnum):
    """Available One Obsession v16 NoobAI SDXL model variants."""

    ONE_OBSESSION_V16_NOOBAI_SDXL = "One_Obsession_v16_NoobAI_SDXL"


class ModelLoader(ForgeModel):
    """One Obsession v16 NoobAI SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.ONE_OBSESSION_V16_NOOBAI_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.ONE_OBSESSION_V16_NOOBAI_SDXL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="One_Obsession_v16_NoobAI_SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the One Obsession v16 NoobAI SDXL pipeline.

        Returns:
            StableDiffusionXLPipeline: The One Obsession v16 NoobAI SDXL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the One Obsession v16 NoobAI SDXL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A beautiful anime girl in a fantasy landscape with colorful flowers"
        ] * batch_size
