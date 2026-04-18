# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ControlNet Canny SDXL model loader implementation
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
    """Available ControlNet Canny SDXL model variants."""

    XINSIR_CONTROLNET_CANNY_SDXL_1_0 = "xinsir_Canny_SDXL_1.0"


class ModelLoader(ForgeModel):
    """ControlNet Canny SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.XINSIR_CONTROLNET_CANNY_SDXL_1_0: ModelConfig(
            pretrained_model_name="xinsir/controlnet-canny-sdxl-1.0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XINSIR_CONTROLNET_CANNY_SDXL_1_0

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
            model="ControlNet Canny SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the ControlNet Canny SDXL pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_controlnet_canny_sdxl_pipe(
            pretrained_model_name, self.base_model
        )

        self.pipeline.to("cpu", dtype=torch.float32)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the UNet model.

        Returns:
            dict: Keyword arguments for the UNet forward method.
        """
        if self.pipeline is None:
            self.load_model()

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

        return {
            "sample": latent_model_input,
            "timestep": timesteps[0],
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
            "down_block_additional_residuals": down_block_additional_residuals,
            "mid_block_additional_residual": mid_block_additional_residual,
        }
