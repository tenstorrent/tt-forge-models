# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ControlNet MLSD SD 1.5 model loader implementation
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
    load_controlnet_mlsd_sd15_pipe,
    create_mlsd_conditioning_image,
    controlnet_mlsd_sd15_preprocessing,
)


class ModelVariant(StrEnum):
    """Available ControlNet MLSD SD 1.5 model variants."""

    CONTROLNET_MLSD_SD15 = "control_v11p_sd15_mlsd"


class ModelLoader(ForgeModel):
    """ControlNet MLSD SD 1.5 model loader implementation."""

    _VARIANTS = {
        ModelVariant.CONTROLNET_MLSD_SD15: ModelConfig(
            pretrained_model_name="lllyasviel/control_v11p_sd15_mlsd",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROLNET_MLSD_SD15

    prompt = "royal chamber with fancy bed"
    base_model = "runwayml/stable-diffusion-v1-5"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ControlNet MLSD SD 1.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the ControlNet MLSD SD 1.5 pipeline and return its UNet.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet module from the pipeline.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_controlnet_mlsd_sd15_pipe(
            pretrained_model_name, self.base_model
        )

        unet = self.pipeline.unet
        if dtype_override is not None:
            unet = unet.to(dtype_override)

        return unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the UNet with ControlNet residuals.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Keyword arguments for the UNet forward pass.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        control_image = create_mlsd_conditioning_image()

        (
            latent_model_input,
            timestep,
            prompt_embeds,
            down_block_additional_residuals,
            mid_block_additional_residual,
        ) = controlnet_mlsd_sd15_preprocessing(
            self.pipeline, self.prompt, control_image
        )

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)
            down_block_additional_residuals = tuple(
                r.to(dtype_override) for r in down_block_additional_residuals
            )
            mid_block_additional_residual = mid_block_additional_residual.to(
                dtype_override
            )

        return {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "down_block_additional_residuals": down_block_additional_residuals,
            "mid_block_additional_residual": mid_block_additional_residual,
        }
