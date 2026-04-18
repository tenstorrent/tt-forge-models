# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Noob SDXL ControlNet Canny model loader implementation
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
from .src.model_utils import (
    load_controlnet_canny_sdxl_pipe,
    create_canny_conditioning_image,
    controlnet_canny_sdxl_preprocessing,
)


class ModelVariant(StrEnum):
    """Available Noob SDXL ControlNet Canny model variants."""

    NOOB_SDXL_CONTROLNET_CANNY = "noob-sdxl-controlnet-canny"


class ModelLoader(ForgeModel):
    """Noob SDXL ControlNet Canny model loader implementation."""

    _VARIANTS = {
        ModelVariant.NOOB_SDXL_CONTROLNET_CANNY: ModelConfig(
            pretrained_model_name="Eugeoter/noob-sdxl-controlnet-canny",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NOOB_SDXL_CONTROLNET_CANNY

    prompt = "A detailed anime illustration, high quality, sharp edges"
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Noob SDXL ControlNet Canny",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_controlnet_canny_sdxl_pipe(
            pretrained_model_name, self.base_model
        )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        control_image = create_canny_conditioning_image()

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
            down_block_additional_residuals,
            mid_block_additional_residual,
        ) = controlnet_canny_sdxl_preprocessing(
            self.pipeline, self.prompt, control_image
        )

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return {
            "sample": latent_model_input,
            "timestep": timesteps[0],
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
            "down_block_additional_residuals": down_block_additional_residuals,
            "mid_block_additional_residual": mid_block_additional_residual,
        }
