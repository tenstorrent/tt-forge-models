# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 3 model loader implementation
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
from .src.model_utils import create_sd3_inputs, load_transformer


class ModelVariant(StrEnum):
    """Available Stable Diffusion 3 model variants."""

    STABLE_DIFFUSION_3_MEDIUM = "3_Medium"


class ModelLoader(ForgeModel):
    """Stable Diffusion 3 model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.STABLE_DIFFUSION_3_MEDIUM: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3-medium-diffusers",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.STABLE_DIFFUSION_3_MEDIUM

    # Shared configuration parameters
    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Stable Diffusion 3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Stable Diffusion 3 transformer with random weights.

        The stabilityai/stable-diffusion-3-medium-diffusers repo is gated, so
        the transformer is created from config for compile-only testing.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            SD3Transformer2DModel: The SD3 transformer instance.
        """
        model = load_transformer()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return synthetic inputs for the SD3 transformer.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors for the transformer:
                - hidden_states (torch.Tensor): Latent input
                - timestep (torch.Tensor): Timestep tensor
                - encoder_hidden_states (torch.Tensor): Text embeddings
                - pooled_projections (torch.Tensor): Pooled text embeddings
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        (
            hidden_states,
            timestep,
            encoder_hidden_states,
            pooled_projections,
        ) = create_sd3_inputs(dtype=dtype)

        return [hidden_states, timestep, encoder_hidden_states, pooled_projections]
