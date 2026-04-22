# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 2.1 model loader implementation
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
    """Available Stable Diffusion 2.1 model variants."""

    BASE = "Base"
    CHARLES_ELENA_STABLE_DIFFUSION_2_1 = "Charles-Elena/stable-diffusion-2-1"


class ModelLoader(ForgeModel):
    """Stable Diffusion 2.1 model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="Manojb/stable-diffusion-2-1-base",
        ),
        ModelVariant.CHARLES_ELENA_STABLE_DIFFUSION_2_1: ModelConfig(
            pretrained_model_name="Charles-Elena/stable-diffusion-2-1",
        ),
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
            variant: Optional variant name string. If None, uses DEFAULT_VARIANT.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Stable Diffusion 2.1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Stable Diffusion 2.1 pipeline and return its UNet component.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            UNet2DConditionModel: The UNet denoising model from the pipeline.
        """
        dtype = dtype_override or torch.bfloat16
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype, **kwargs
        )
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Stable Diffusion 2.1 UNet model.

        Args:
            dtype_override: Optional torch.dtype to override input dtype.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Keyword arguments for the UNet forward method.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override or torch.bfloat16
        prompt = ["a photo of an astronaut riding a horse on mars"] * batch_size

        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.pipeline.text_encoder(text_inputs.input_ids)[0]

        unet = self.pipeline.unet
        height, width = 512, 512
        latents = torch.randn(
            (batch_size, unet.config.in_channels, height // 8, width // 8),
            dtype=dtype,
        )

        self.pipeline.scheduler.set_timesteps(1)
        timestep = self.pipeline.scheduler.timesteps[0]

        return {
            "sample": latents,
            "timestep": timestep,
            "encoder_hidden_states": text_embeddings.to(dtype),
        }
