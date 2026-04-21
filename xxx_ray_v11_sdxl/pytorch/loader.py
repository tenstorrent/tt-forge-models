# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XXX-Ray v11 SDXL (John6666/xxx-ray-v11-sdxl) model loader implementation.

XXX-Ray v11 SDXL is a realistic/photorealistic Stable Diffusion XL
checkpoint for text-to-image generation.

Available variants:
- XXX_RAY_V11_SDXL: John6666/xxx-ray-v11-sdxl text-to-image generation
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


REPO_ID = "John6666/xxx-ray-v11-sdxl"


class ModelVariant(StrEnum):
    """Available XXX-Ray v11 SDXL model variants."""

    XXX_RAY_V11_SDXL = "xxx-ray-v11-sdxl"


class ModelLoader(ForgeModel):
    """XXX-Ray v11 SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.XXX_RAY_V11_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.XXX_RAY_V11_SDXL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="xxx-ray-v11-sdxl",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the XXX-Ray v11 SDXL pipeline.

        Returns:
            StableDiffusionXLPipeline: The XXX-Ray v11 SDXL pipeline instance.
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
        """Load and return sample text prompts for the XXX-Ray v11 SDXL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
