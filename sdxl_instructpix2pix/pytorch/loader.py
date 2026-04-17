# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SDXL InstructPix2Pix model loader implementation
"""

from typing import Optional

import torch

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
from .src.model_utils import (
    create_dummy_input_image,
    load_sdxl_instructpix2pix_pipe,
    sdxl_instructpix2pix_preprocessing,
)


class ModelVariant(StrEnum):
    """Available SDXL InstructPix2Pix model variants."""

    SDXL_INSTRUCTPIX2PIX_768 = "sdxl-instructpix2pix-768"


class ModelLoader(ForgeModel):
    """SDXL InstructPix2Pix model loader implementation."""

    _VARIANTS = {
        ModelVariant.SDXL_INSTRUCTPIX2PIX_768: ModelConfig(
            pretrained_model_name="diffusers/sdxl-instructpix2pix-768",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SDXL_INSTRUCTPIX2PIX_768

    prompt = "Turn sky into a cloudy one"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SDXL InstructPix2Pix",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the SDXL InstructPix2Pix pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_sdxl_instructpix2pix_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the SDXL InstructPix2Pix UNet.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Keyword arguments for the UNet forward method.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        input_image = create_dummy_input_image()

        (
            scaled_latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
        ) = sdxl_instructpix2pix_preprocessing(self.pipeline, self.prompt, input_image)

        timestep = timesteps[0]

        if dtype_override:
            scaled_latent_model_input = scaled_latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return {
            "sample": scaled_latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
        }
