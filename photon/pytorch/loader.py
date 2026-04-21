# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Photon v1 model loader implementation
"""

import torch
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
from typing import Optional


class ModelVariant(StrEnum):
    """Available Photon model variants."""

    V1 = "v1"
    SAM749_V1 = "sam749_v1"


class ModelLoader(ForgeModel):
    """Photon model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.V1: ModelConfig(
            pretrained_model_name="digiplay/Photon_v1",
        ),
        ModelVariant.SAM749_V1: ModelConfig(
            pretrained_model_name="sam749/Photon-v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="Photon",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Photon v1 UNet from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            UNet2DConditionModel: The UNet backbone used for denoising.
        """
        dtype = dtype_override or torch.bfloat16
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype, **kwargs
        )
        if dtype_override is not None:
            self.pipe.unet = self.pipe.unet.to(dtype_override)
        return self.pipe.unet

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Load and return sample inputs for the Photon v1 UNet.

        Args:
            dtype_override: Optional torch.dtype for input tensors.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Input tensors for the UNet forward pass.
        """
        if self.pipe is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override or torch.bfloat16

        prompt = "a photo of an astronaut riding a horse on mars"
        text_inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        encoder_hidden_states = self.pipe.text_encoder(text_inputs.input_ids)[0].to(
            dtype=dtype
        )

        vae_scale_factor = 2 ** (len(self.pipe.vae.config.block_out_channels) - 1)
        latent_size = 512 // vae_scale_factor
        num_channels = self.pipe.unet.config.in_channels

        sample = torch.randn(
            batch_size,
            num_channels,
            latent_size,
            latent_size,
            dtype=dtype,
        )
        timestep = torch.tensor([1], dtype=dtype)

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
