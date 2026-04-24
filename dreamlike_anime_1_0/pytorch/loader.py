# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Dreamlike Anime 1.0 model loader implementation
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
    """Available Dreamlike Anime 1.0 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Dreamlike Anime 1.0 model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="dreamlike-art/dreamlike-anime-1.0",
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
            variant: Optional variant name string. If None, uses DEFAULT_VARIANT.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Dreamlike Anime 1.0",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Dreamlike Anime 1.0 pipeline and return its UNet.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            torch.nn.Module: The UNet component of the pipeline.
        """
        dtype = dtype_override or torch.bfloat16
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype, **kwargs
        )
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Load preprocessed tensor inputs for the Dreamlike Anime 1.0 UNet.

        Args:
            dtype_override: Optional torch.dtype for input tensors.
            batch_size: Number of samples in the batch.

        Returns:
            list: [sample, timestep, encoder_hidden_states] tensors for the UNet.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override or torch.bfloat16
        pipe = self.pipeline

        prompt = [
            "anime, masterpiece, high quality, 1girl, solo, long hair, looking at viewer, blush, smile, blue eyes, in a colorful flower field",
        ] * batch_size

        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids)[0].to(
                dtype
            )

        height, width = 512, 512
        latent_height = height // pipe.vae_scale_factor
        latent_width = width // pipe.vae_scale_factor
        in_channels = pipe.unet.config.in_channels
        sample = torch.randn(
            batch_size, in_channels, latent_height, latent_width, dtype=dtype
        )

        timestep = torch.tensor([1], dtype=torch.long)

        return [sample, timestep, encoder_hidden_states]
