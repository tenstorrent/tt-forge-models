# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Anything V5 (genai-archive) model loader implementation
"""

import torch
from typing import Optional

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
from diffusers import StableDiffusionPipeline


class ModelVariant(StrEnum):
    """Available Anything V5 (genai-archive) model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Anything V5 (genai-archive) model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="genai-archive/anything-v5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

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
            model="Anything V5 (genai-archive)",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Anything V5 UNet from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            UNet2DConditionModel: The pre-trained UNet from the Anything V5 pipeline.
        """
        dtype = dtype_override or torch.bfloat16
        pipe = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype, **kwargs
        )
        self.scheduler = pipe.scheduler
        model = pipe.unet
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Anything V5 UNet.

        Args:
            dtype_override: Optional torch.dtype for input tensors.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary with sample latents, timestep, and encoder hidden states.
        """
        dtype = dtype_override or torch.bfloat16
        # SD1.x UNet: 4 latent channels, 64x64 spatial (for 512x512 output), 77-token CLIP embeddings
        sample = torch.randn((batch_size, 4, 64, 64), dtype=dtype)
        timestep = torch.tensor([1])
        encoder_hidden_states = torch.randn((batch_size, 77, 768), dtype=dtype)
        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
