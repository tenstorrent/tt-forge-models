# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ControlNet Union ProMax SDXL model loader implementation
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
    load_controlnet_union_promax_sdxl_pipe,
    create_union_conditioning_image,
    controlnet_union_promax_sdxl_preprocessing,
)


class ModelVariant(StrEnum):
    """Available ControlNet Union ProMax SDXL model variants."""

    CONTROLNET_UNION_PROMAX_SDXL_1_0 = "Union_ProMax_SDXL_1.0"
    CONTROLNET_UNION_PROMAX_SDXL_1_0_BRAD_TWINKL = "Union_ProMax_SDXL_1.0_brad_twinkl"


class ModelLoader(ForgeModel):
    """ControlNet Union ProMax SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.CONTROLNET_UNION_PROMAX_SDXL_1_0: ModelConfig(
            pretrained_model_name="OzzyGT/controlnet-union-promax-sdxl-1.0",
        ),
        ModelVariant.CONTROLNET_UNION_PROMAX_SDXL_1_0_BRAD_TWINKL: ModelConfig(
            pretrained_model_name="brad-twinkl/controlnet-union-sdxl-1.0-promax",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROLNET_UNION_PROMAX_SDXL_1_0

    prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ControlNet Union ProMax SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the ControlNet Union ProMax SDXL pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_controlnet_union_promax_sdxl_pipe(
            pretrained_model_name, self.base_model
        )

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)
        else:
            self.pipeline.unet = self.pipeline.unet.float()

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ControlNet Union ProMax SDXL UNet.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Keyword arguments for the UNet forward method.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        control_image = create_union_conditioning_image()

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
            down_block_additional_residuals,
            mid_block_additional_residual,
        ) = controlnet_union_promax_sdxl_preprocessing(
            self.pipeline, self.prompt, control_image
        )

        timestep = timesteps[0]

        target_dtype = dtype_override if dtype_override else torch.float32
        latent_model_input = latent_model_input.to(target_dtype)
        timestep = timestep.to(target_dtype)
        prompt_embeds = prompt_embeds.to(target_dtype)
        added_cond_kwargs = {
            k: v.to(target_dtype) if isinstance(v, torch.Tensor) else v
            for k, v in added_cond_kwargs.items()
        }
        down_block_additional_residuals = tuple(
            r.to(target_dtype) for r in down_block_additional_residuals
        )
        mid_block_additional_residual = mid_block_additional_residual.to(target_dtype)

        return {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
            "down_block_additional_residuals": down_block_additional_residuals,
            "mid_block_additional_residual": mid_block_additional_residual,
        }
