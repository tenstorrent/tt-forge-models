# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AuraFlow (fal/AuraFlow-v0.3) model loader implementation.

AuraFlow is a flow-based text-to-image generation model using the
AuraFlowPipeline from diffusers.

Available variants:
- V0_3: fal/AuraFlow-v0.3 text-to-image generation
"""

from typing import Optional

import torch
from diffusers import AuraFlowPipeline

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


class ModelVariant(StrEnum):
    """Available AuraFlow model variants."""

    V0_3 = "v0.3"


class ModelLoader(ForgeModel):
    """AuraFlow model loader implementation."""

    _VARIANTS = {
        ModelVariant.V0_3: ModelConfig(
            pretrained_model_name="fal/AuraFlow-v0.3",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.V0_3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AuraFlow",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the AuraFlow transformer as a torch.nn.Module.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The AuraFlow transformer model.
        """
        dtype = dtype_override or torch.float16
        self.pipeline = AuraFlowPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return synthetic inputs for the AuraFlow transformer.

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Optional batch size.

        Returns:
            dict: Input tensors for the transformer forward pass.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override or torch.float16
        config = self.pipeline.transformer.config

        # Small latent dimensions for testing (must be divisible by patch_size=2)
        latent_height = 8
        latent_width = 8

        hidden_states = torch.randn(
            batch_size,
            config.in_channels,
            latent_height,
            latent_width,
            dtype=dtype,
        )
        encoder_hidden_states = torch.randn(
            batch_size,
            256,
            config.joint_attention_dim,
            dtype=dtype,
        )
        timestep = torch.tensor([1.0] * batch_size, dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "return_dict": False,
        }
