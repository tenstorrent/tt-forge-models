# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Noob SDXL ControlNet Tile model loader implementation
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
    load_controlnet_tile_sdxl_pipe,
    create_tile_conditioning_image,
    controlnet_tile_sdxl_preprocessing,
)


class ModelVariant(StrEnum):
    """Available Noob SDXL ControlNet Tile model variants."""

    NOOB_SDXL_CONTROLNET_TILE = "noob-sdxl-controlnet-tile"


class ModelLoader(ForgeModel):
    """Noob SDXL ControlNet Tile model loader implementation."""

    _VARIANTS = {
        ModelVariant.NOOB_SDXL_CONTROLNET_TILE: ModelConfig(
            pretrained_model_name="Eugeoter/noob-sdxl-controlnet-tile",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NOOB_SDXL_CONTROLNET_TILE

    prompt = "A detailed anime illustration, high quality, vivid colors"
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Noob SDXL ControlNet Tile",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the Noob SDXL ControlNet Tile pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            UNet2DConditionModel: The UNet component of the pipeline.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_controlnet_tile_sdxl_pipe(
            pretrained_model_name, self.base_model
        )

        # Keep the pipeline in float32 so preprocessing (controlnet forward, text encoding)
        # runs without dtype mismatches. Only the UNet is converted for inference.
        if dtype_override is not None:
            return self.pipeline.unet.to(dtype_override)
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Noob SDXL ControlNet Tile model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            Dict: Keyword arguments for UNet forward with ControlNet residuals.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        control_image = create_tile_conditioning_image()

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
            down_block_additional_residuals,
            mid_block_additional_residual,
        ) = controlnet_tile_sdxl_preprocessing(
            self.pipeline, self.prompt, control_image
        )

        # The UNet expects a single timestep, not the full scheduler schedule.
        timestep = timesteps[0]

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)
            added_cond_kwargs = {
                k: v.to(dtype_override) for k, v in added_cond_kwargs.items()
            }
            down_block_additional_residuals = tuple(
                t.to(dtype_override) for t in down_block_additional_residuals
            )
            mid_block_additional_residual = mid_block_additional_residual.to(
                dtype_override
            )

        return {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
            "down_block_additional_residuals": down_block_additional_residuals,
            "mid_block_additional_residual": mid_block_additional_residual,
        }
