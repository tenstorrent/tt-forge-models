# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ControlNet OpenPose SDXL model loader implementation
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
    load_controlnet_openpose_sdxl_pipe,
    create_openpose_conditioning_image,
    controlnet_openpose_sdxl_preprocessing,
)


class ModelVariant(StrEnum):
    """Available ControlNet OpenPose SDXL model variants."""

    CONTROLNET_OPENPOSE_SDXL_1_0 = "OpenPose_SDXL_1.0"
    XINSIR_CONTROLNET_OPENPOSE_SDXL_1_0 = "xinsir_OpenPose_SDXL_1.0"


class ModelLoader(ForgeModel):
    """ControlNet OpenPose SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.CONTROLNET_OPENPOSE_SDXL_1_0: ModelConfig(
            pretrained_model_name="thibaud/controlnet-openpose-sdxl-1.0",
        ),
        ModelVariant.XINSIR_CONTROLNET_OPENPOSE_SDXL_1_0: ModelConfig(
            pretrained_model_name="xinsir/controlnet-openpose-sdxl-1.0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROLNET_OPENPOSE_SDXL_1_0

    prompt = "A person dancing in a colorful room, high quality"
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ControlNet OpenPose SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the ControlNet OpenPose SDXL pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_controlnet_openpose_sdxl_pipe(
            pretrained_model_name, self.base_model
        )

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ControlNet OpenPose SDXL UNet.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Keyword arguments for the UNet forward method.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        control_image = create_openpose_conditioning_image()

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
            down_block_additional_residuals,
            mid_block_additional_residual,
        ) = controlnet_openpose_sdxl_preprocessing(
            self.pipeline, self.prompt, control_image
        )

        timestep = timesteps[0]

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)
            added_cond_kwargs = {
                k: v.to(dtype_override) if isinstance(v, torch.Tensor) else v
                for k, v in added_cond_kwargs.items()
            }
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
            "added_cond_kwargs": added_cond_kwargs,
            "down_block_additional_residuals": down_block_additional_residuals,
            "mid_block_additional_residual": mid_block_additional_residual,
        }
