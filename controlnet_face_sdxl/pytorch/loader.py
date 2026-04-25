# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ControlNet Face SDXL model loader implementation
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
    load_controlnet_face_sdxl_pipe,
    create_face_conditioning_image,
    controlnet_face_sdxl_preprocessing,
)


class ModelVariant(StrEnum):
    """Available ControlNet Face SDXL model variants."""

    CONTROLNET_FACE_SDXL = "Face_SDXL"


class ModelLoader(ForgeModel):
    """ControlNet Face SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.CONTROLNET_FACE_SDXL: ModelConfig(
            pretrained_model_name="okaris/face-controlnet-xl",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROLNET_FACE_SDXL

    prompt = "A portrait of a person with a warm smile, high quality"
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ControlNet Face SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ControlNet Face SDXL UNet.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            UNet2DConditionModel: The UNet module from the pipeline.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_controlnet_face_sdxl_pipe(
            pretrained_model_name, self.base_model
        )

        unet = self.pipeline.unet
        if dtype_override is not None:
            unet = unet.to(dtype_override)

        unet.eval()
        return unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ControlNet Face SDXL UNet.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Keyword arguments for UNet2DConditionModel.forward():
                - sample (torch.Tensor)
                - timestep (torch.Tensor)
                - encoder_hidden_states (torch.Tensor)
                - added_cond_kwargs (dict)
                - down_block_additional_residuals (tuple of torch.Tensor)
                - mid_block_additional_residual (torch.Tensor)
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        control_image = create_face_conditioning_image()

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
            down_block_additional_residuals,
            mid_block_additional_residual,
        ) = controlnet_face_sdxl_preprocessing(
            self.pipeline, self.prompt, control_image
        )

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
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
            "timestep": timesteps[0],
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
            "down_block_additional_residuals": down_block_additional_residuals,
            "mid_block_additional_residual": mid_block_additional_residual,
        }
