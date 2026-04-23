# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ControlNet Depth Zoe SDXL model loader implementation
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
    load_controlnet_depth_zoe_sdxl_pipe,
    create_depth_conditioning_image,
    controlnet_depth_zoe_sdxl_preprocessing,
)


class ModelVariant(StrEnum):
    """Available ControlNet Depth Zoe SDXL model variants."""

    ZOE_DEPTH_CONTROLNET_XL = "Zoe_Depth_ControlNet_XL"


class ModelLoader(ForgeModel):
    """ControlNet Depth Zoe SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.ZOE_DEPTH_CONTROLNET_XL: ModelConfig(
            pretrained_model_name="okaris/zoe-depth-controlnet-xl",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ZOE_DEPTH_CONTROLNET_XL

    prompt = "A scenic landscape with mountains and a river, high quality"
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ControlNet Depth Zoe SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ControlNet Depth Zoe SDXL pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet model from the pipeline.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_controlnet_depth_zoe_sdxl_pipe(
            pretrained_model_name, self.base_model
        )

        if dtype_override is not None:
            self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ControlNet Depth Zoe SDXL model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            Dict: Keyword arguments for the UNet forward with ControlNet residuals.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        control_image = create_depth_conditioning_image()

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs,
            down_block_additional_residuals,
            mid_block_additional_residual,
        ) = controlnet_depth_zoe_sdxl_preprocessing(
            self.pipeline, self.prompt, control_image
        )

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)
            added_cond_kwargs = {
                k: v.to(dtype_override) if isinstance(v, torch.Tensor) else v
                for k, v in added_cond_kwargs.items()
            }
            down_block_additional_residuals = [
                r.to(dtype_override) for r in down_block_additional_residuals
            ]
            mid_block_additional_residual = mid_block_additional_residual.to(
                dtype_override
            )

        return {
            "sample": latent_model_input,
            "timestep": timesteps,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
            "down_block_additional_residuals": down_block_additional_residuals,
            "mid_block_additional_residual": mid_block_additional_residual,
        }
