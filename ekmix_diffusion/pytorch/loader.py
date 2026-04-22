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
        self.pipeline = None

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
        """Load the Ekmix Diffusion pipeline and return its UNet.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            torch.nn.Module: The UNet denoising model from the pipeline.
        """
        dtype = dtype_override or torch.bfloat16
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype, **kwargs
        )
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the UNet model.

        Args:
            dtype_override: Optional dtype override.
            batch_size: Optional batch size.

        Returns:
            dict: Keyword arguments for the UNet forward method.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override or torch.bfloat16
        pipe = self.pipeline
        unet = pipe.unet

        prompt = "masterpiece, best quality, 1girl, long hair, solo, looking at viewer"
        max_length = min(pipe.tokenizer.model_max_length, 77)
        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids)[0].to(
                dtype
            )

        in_channels = unet.config.in_channels
        sample_size = unet.config.sample_size
        latent_sample = torch.randn(
            batch_size, in_channels, sample_size, sample_size, dtype=dtype
        )
        timestep = torch.tensor([1], dtype=torch.long)

        return {
            "sample": latent_sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
