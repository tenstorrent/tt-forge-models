# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ekmix Diffusion model loader implementation
"""

import torch
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available Ekmix Diffusion model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Ekmix Diffusion model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="EK12317/Ekmix-Diffusion",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional variant. If None, uses default.

        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="Ekmix Diffusion",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Ekmix Diffusion pipeline from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            StableDiffusionPipeline: The pre-trained Ekmix Diffusion pipeline object.
        """
        dtype = dtype_override or torch.bfloat16
        pipe = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype, **kwargs
        )
        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Ekmix Diffusion model.

        Args:
            dtype_override: This parameter is ignored for this model.
            batch_size: Optional batch size for the prompts.

        Returns:
            list: A list of sample text prompts.
        """

        prompt = [
            "masterpiece, best quality, 1girl, long hair, solo, looking at viewer",
        ] * batch_size
        return prompt
