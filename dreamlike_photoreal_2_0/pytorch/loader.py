# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Dreamlike Photoreal 2.0 model loader implementation
"""

from typing import Optional

import torch
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Dreamlike Photoreal 2.0 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Dreamlike Photoreal 2.0 model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="dreamlike-art/dreamlike-photoreal-2.0",
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
            variant: Optional variant name string. If None, uses DEFAULT_VARIANT.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Dreamlike Photoreal 2.0",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Dreamlike Photoreal 2.0 UNet from the pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            UNet2DConditionModel: The pre-trained UNet model.
        """
        dtype = dtype_override or torch.bfloat16
        pipe = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype, **kwargs
        )
        self._tokenizer = pipe.tokenizer
        self._text_encoder = pipe.text_encoder
        self._scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        unet = pipe.unet
        self._in_channels = unet.in_channels
        return unet

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Load and return sample inputs for the Dreamlike Photoreal 2.0 UNet.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing sample, timestep, and encoder_hidden_states.
        """
        dtype = dtype_override or torch.bfloat16

        prompt = [
            "photo, a church in the middle of a field of crops, bright cinematic lighting, gopro, fisheye lens",
        ] * batch_size
        text_input = self._tokenizer(prompt, return_tensors="pt")
        text_embeddings = self._text_encoder(text_input.input_ids)[0]

        height, width = 512, 512
        latents = torch.randn((batch_size, self._in_channels, height // 8, width // 8))

        num_inference_steps = 1
        self._scheduler.set_timesteps(num_inference_steps)
        latents = latents * self._scheduler.init_noise_sigma
        latent_model_input = self._scheduler.scale_model_input(latents, 0)

        return {
            "sample": latent_model_input.to(dtype),
            "timestep": 0,
            "encoder_hidden_states": text_embeddings.to(dtype),
        }
