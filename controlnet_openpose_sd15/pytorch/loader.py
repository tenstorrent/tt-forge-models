# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ControlNet OpenPose SD1.5 model loader implementation
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
    load_controlnet_openpose_sd15_pipe,
    create_openpose_conditioning_image,
    controlnet_openpose_sd15_preprocessing,
)


class ModelVariant(StrEnum):
    """Available ControlNet OpenPose SD1.5 model variants."""

    CONTROLNET_OPENPOSE_SD15 = "OpenPose_SD1.5"
    SD_CONTROLNET_OPENPOSE = "SD_ControlNet_OpenPose"


class ModelLoader(ForgeModel):
    """ControlNet OpenPose SD1.5 model loader implementation."""

    _VARIANTS = {
        ModelVariant.CONTROLNET_OPENPOSE_SD15: ModelConfig(
            pretrained_model_name="lllyasviel/control_v11p_sd15_openpose",
        ),
        ModelVariant.SD_CONTROLNET_OPENPOSE: ModelConfig(
            pretrained_model_name="lllyasviel/sd-controlnet-openpose",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROLNET_OPENPOSE_SD15

    prompt = "A person dancing in a colorful room, high quality"
    base_model = "runwayml/stable-diffusion-v1-5"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ControlNet OpenPose SD1.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ControlNet OpenPose SD1.5 pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet module.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_controlnet_openpose_sd15_pipe(
            pretrained_model_name, self.base_model
        )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ControlNet OpenPose SD1.5 model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            Dict: Keyword arguments for the UNet forward pass.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        control_image = create_openpose_conditioning_image()

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            down_block_additional_residuals,
            mid_block_additional_residual,
        ) = controlnet_openpose_sd15_preprocessing(
            self.pipeline, self.prompt, control_image
        )

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return {
            "sample": latent_model_input,
            "timestep": timesteps[0],
            "encoder_hidden_states": prompt_embeds,
            "down_block_additional_residuals": down_block_additional_residuals,
            "mid_block_additional_residual": mid_block_additional_residual,
        }
