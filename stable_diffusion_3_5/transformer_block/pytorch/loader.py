# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 3.5 model loader implementation
https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
"""

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
import torch
from diffusers.models import SD3Transformer2DModel
from typing import Optional


class ModelVariant(StrEnum):
    """Available Stable Diffusion 3.5 transformer block variants."""

    MEDIUM_TRANSFORMER_BLOCK = "medium-transformer-block"


class ModelLoader(ForgeModel):
    """Stable Diffusion 3.5 transformer block model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.MEDIUM_TRANSFORMER_BLOCK: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3.5-medium",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MEDIUM_TRANSFORMER_BLOCK

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
        return ModelInfo(
            model="stable_diffusion_3_5",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the first transformer block from Stable Diffusion 3.5.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            The first transformer block from the SD3 transformer model.
        """
        model_path = self._variant_config.pretrained_model_name

        transformer_kwargs = {}
        if dtype_override is not None:
            transformer_kwargs["torch_dtype"] = dtype_override

        transformer = SD3Transformer2DModel.from_pretrained(
            model_path, subfolder="transformer", **transformer_kwargs
        )
        return transformer.transformer_blocks[0]

    def load_inputs(self, batch_size=1, dtype_override=None):
        """Load and return sample inputs for the Stable Diffusion 3.5 transformer block.

        Args:
            dtype_override: Optional torch.dtype to override the default dtype of the inputs.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing transformer block inputs.
        """
        hidden_states = torch.randn(batch_size, 1024, 1536)
        encoder_hidden_states = torch.randn(batch_size, 333, 1536)
        temb = torch.rand(batch_size, 1536)
        if dtype_override is not None:
            hidden_states = hidden_states.to(dtype_override)
            encoder_hidden_states = encoder_hidden_states.to(dtype_override)
            temb = temb.to(dtype_override)
        joint_attention_kwargs = {}

        arguments = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "temb": temb,
            "joint_attention_kwargs": joint_attention_kwargs,
        }
        return arguments
