# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Noob SDXL ControlNet Scribble PidiNet model loader implementation
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
    load_controlnet_scribble_pidinet_sdxl_pipe,
    create_scribble_pidinet_conditioning_image,
    controlnet_scribble_pidinet_sdxl_preprocessing,
)


class ModelVariant(StrEnum):
    """Available Noob SDXL ControlNet Scribble PidiNet model variants."""

    NOOB_SDXL_CONTROLNET_SCRIBBLE_PIDINET = "noob-sdxl-controlnet-scribble_pidinet"


class ModelLoader(ForgeModel):
    """Noob SDXL ControlNet Scribble PidiNet model loader implementation."""

    _VARIANTS = {
        ModelVariant.NOOB_SDXL_CONTROLNET_SCRIBBLE_PIDINET: ModelConfig(
            pretrained_model_name="Eugeoter/noob-sdxl-controlnet-scribble_pidinet",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NOOB_SDXL_CONTROLNET_SCRIBBLE_PIDINET

    prompt = "A detailed anime illustration from a scribble sketch, high quality"
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Noob SDXL ControlNet Scribble PidiNet",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Noob SDXL ControlNet Scribble PidiNet pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_controlnet_scribble_pidinet_sdxl_pipe(
            pretrained_model_name, self.base_model
        )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Noob SDXL ControlNet Scribble PidiNet model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Keyword arguments for the UNet forward method:
                - sample (torch.Tensor): Latent input for the UNet
                - timestep (torch.Tensor): Single timestep tensor
                - encoder_hidden_states (torch.Tensor): Encoded prompt embeddings
                - added_cond_kwargs (dict): Additional conditioning inputs
                - down_block_additional_residuals (tuple of torch.Tensor): ControlNet residuals
                - mid_block_additional_residual (torch.Tensor): ControlNet mid residual
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        control_image = create_scribble_pidinet_conditioning_image()

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
            down_block_additional_residuals,
            mid_block_additional_residual,
        ) = controlnet_scribble_pidinet_sdxl_preprocessing(
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

        return {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
            "down_block_additional_residuals": down_block_additional_residuals,
            "mid_block_additional_residual": mid_block_additional_residual,
        }
