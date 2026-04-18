# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ControlNet IP2P SD1.5 model loader implementation
"""

from typing import Any, Optional

import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from .src.model_utils import (
    controlnet_ip2p_sd15_preprocessing,
    create_ip2p_conditioning_image,
    load_controlnet_ip2p_sd15_pipe,
)


class ModelVariant(StrEnum):
    """Available ControlNet IP2P SD1.5 model variants."""

    LLLYASVIEL_CONTROL_V11E_SD15_IP2P = "lllyasviel_control_v11e_sd15_ip2p"


class ModelLoader(ForgeModel):
    """ControlNet IP2P SD1.5 model loader implementation."""

    _VARIANTS = {
        ModelVariant.LLLYASVIEL_CONTROL_V11E_SD15_IP2P: ModelConfig(
            pretrained_model_name="lllyasviel/control_v11e_sd15_ip2p",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLLYASVIEL_CONTROL_V11E_SD15_IP2P

    prompt = "make it on fire"
    base_model = "runwayml/stable-diffusion-v1-5"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ControlNet IP2P SD1.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_controlnet_ip2p_sd15_pipe(
            pretrained_model_name, self.base_model
        )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        control_image = create_ip2p_conditioning_image()

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            down_block_additional_residuals,
            mid_block_additional_residual,
        ) = controlnet_ip2p_sd15_preprocessing(
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

    def unpack_forward_output(self, fwd_output: Any) -> torch.Tensor:
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        if hasattr(fwd_output, "sample"):
            return fwd_output.sample
        return fwd_output
