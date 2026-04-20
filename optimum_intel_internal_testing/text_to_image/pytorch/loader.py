# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Optimum Intel Internal Testing tiny-stable-diffusion-with-textual-inversion
model loader implementation for text-to-image generation.
"""

import torch
from diffusers import StableDiffusionPipeline
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available tiny-stable-diffusion-with-textual-inversion model variants."""

    TINY_STABLE_DIFFUSION_WITH_TEXTUAL_INVERSION = (
        "tiny_stable_diffusion_with_textual_inversion"
    )


class ModelLoader(ForgeModel):
    """Optimum Intel Internal Testing tiny-stable-diffusion-with-textual-inversion loader."""

    _VARIANTS = {
        ModelVariant.TINY_STABLE_DIFFUSION_WITH_TEXTUAL_INVERSION: ModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-stable-diffusion-with-textual-inversion",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_STABLE_DIFFUSION_WITH_TEXTUAL_INVERSION

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Optimum_Intel_Internal_Testing",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the tiny Stable Diffusion pipeline from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            StableDiffusionPipeline: The pre-trained Stable Diffusion pipeline object.
        """
        dtype = dtype_override or torch.bfloat16
        pipe = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype, **kwargs
        )
        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the tiny Stable Diffusion model.

        Args:
            dtype_override: This parameter is ignored for this model.
            batch_size: Optional batch size for the prompts.

        Returns:
            list: A list of sample text prompts.
        """
        prompt = [
            "a photo of an astronaut riding a horse on mars",
        ] * batch_size
        return prompt
