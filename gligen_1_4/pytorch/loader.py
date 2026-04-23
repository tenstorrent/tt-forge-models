# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLIGEN 1.4 model loader implementation

GLIGEN extends Stable Diffusion with grounded text-to-image generation,
allowing placement of objects at specified bounding box locations.

Reference: https://huggingface.co/masterful/gligen-1-4-generation-text-box
"""

from typing import Optional

import torch
from diffusers import StableDiffusionGLIGENPipeline

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
    """Available GLIGEN model variants."""

    GENERATION_TEXT_BOX = "generation-text-box"


class ModelLoader(ForgeModel):
    """GLIGEN 1.4 model loader implementation."""

    _VARIANTS = {
        ModelVariant.GENERATION_TEXT_BOX: ModelConfig(
            pretrained_model_name="masterful/gligen-1-4-generation-text-box",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GENERATION_TEXT_BOX

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GLIGEN 1.4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the GLIGEN pipeline and return the UNet as a torch.nn.Module.

        Returns:
            torch.nn.Module: The UNet component of the GLIGEN pipeline.
        """
        dtype = dtype_override or torch.bfloat16
        self.pipeline = StableDiffusionGLIGENPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load preprocessed tensor inputs for the GLIGEN UNet.

        Returns:
            list: [sample, timestep, encoder_hidden_states] tensors for the UNet.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override or torch.bfloat16
        pipe = self.pipeline

        prompt = [
            "a waterfall and a modern high speed train in a beautiful forest with fall foliage",
        ] * batch_size

        # Encode text prompt
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

        # Create dummy latent (SD1.x uses 64x64 latent for 512x512 images)
        height, width = 512, 512
        latent_height = height // pipe.vae_scale_factor
        latent_width = width // pipe.vae_scale_factor
        in_channels = pipe.unet.config.in_channels
        sample = torch.randn(
            batch_size, in_channels, latent_height, latent_width, dtype=dtype
        )

        # Single timestep
        timestep = torch.tensor([1], dtype=torch.long)

        return [sample, timestep, encoder_hidden_states]
