# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LCM-LoRA SDXL model loader implementation.

LCM-LoRA (Latent Consistency Model LoRA) is a distilled consistency adapter for
SDXL that reduces inference from 20-50 steps down to 2-8 steps while maintaining
image quality.

Available variants:
- LCM_LORA_SDXL: latent-consistency/lcm-lora-sdxl text-to-image generation
"""

from typing import Any, Optional

import torch

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
from .src.model_utils import load_pipe, lcm_lora_sdxl_preprocessing


class ModelVariant(StrEnum):
    """Available LCM-LoRA SDXL model variants."""

    LCM_LORA_SDXL = "LCM_LoRA_SDXL"


class ModelLoader(ForgeModel):
    """LCM-LoRA SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.LCM_LORA_SDXL: ModelConfig(
            pretrained_model_name="latent-consistency/lcm-lora-sdxl",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.LCM_LORA_SDXL

    prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LCM-LoRA SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the LCM-LoRA SDXL pipeline.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.pipeline = load_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, **kwargs):
        """Load and return sample inputs for the LCM-LoRA SDXL UNet.

        Returns:
            dict: Keyword arguments for the UNet forward method.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            _timestep_cond,
            added_cond_kwargs,
            _add_time_ids,
        ) = lcm_lora_sdxl_preprocessing(self.pipeline, self.prompt)

        timestep = timesteps[0]

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
        }

    def unpack_forward_output(self, fwd_output: Any) -> torch.Tensor:
        """Unpack UNet output to the sample tensor."""
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        if hasattr(fwd_output, "sample"):
            return fwd_output.sample
        return fwd_output
