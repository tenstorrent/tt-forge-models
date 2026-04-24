# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ControlNet Scribble SD1.5 model loader implementation
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
    load_controlnet_scribble_sd15_pipe,
    create_scribble_conditioning_image,
    controlnet_scribble_sd15_preprocessing,
)


class ModelVariant(StrEnum):
    """Available ControlNet Scribble SD1.5 model variants."""

    LLLYASVIEL_CONTROL_V11P_SD15_SCRIBBLE = "lllyasviel_control_v11p_sd15_scribble"
    LLLYASVIEL_SD_CONTROLNET_SCRIBBLE = "lllyasviel_sd_controlnet_scribble"


class ModelLoader(ForgeModel):
    """ControlNet Scribble SD1.5 model loader implementation."""

    _VARIANTS = {
        ModelVariant.LLLYASVIEL_CONTROL_V11P_SD15_SCRIBBLE: ModelConfig(
            pretrained_model_name="lllyasviel/control_v11p_sd15_scribble",
        ),
        ModelVariant.LLLYASVIEL_SD_CONTROLNET_SCRIBBLE: ModelConfig(
            pretrained_model_name="lllyasviel/sd-controlnet-scribble",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLLYASVIEL_CONTROL_V11P_SD15_SCRIBBLE

    prompt = "michael jackson concert"
    base_model = "runwayml/stable-diffusion-v1-5"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ControlNet Scribble SD1.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the ControlNet Scribble SD1.5 pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_controlnet_scribble_sd15_pipe(
            pretrained_model_name, self.base_model
        )

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ControlNet Scribble SD1.5 UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Keyword arguments for the UNet forward method:
                - sample (torch.Tensor): Latent input for the UNet
                - timestep (torch.Tensor): Single timestep tensor
                - encoder_hidden_states (torch.Tensor): Encoded prompt embeddings
                - down_block_additional_residuals (tuple of torch.Tensor): ControlNet residuals
                - mid_block_additional_residual (torch.Tensor): ControlNet mid-block residual
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        control_image = create_scribble_conditioning_image()

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            down_block_additional_residuals,
            mid_block_additional_residual,
        ) = controlnet_scribble_sd15_preprocessing(
            self.pipeline, self.prompt, control_image
        )

        timestep = timesteps[0]

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
