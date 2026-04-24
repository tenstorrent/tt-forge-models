# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Garage Mix Noob V-Pred Eps v1.0 V-Pred SDXL
(John6666/garage-mix-noob-vpred-eps-v10-vpred-sdxl) model loader implementation.

Garage Mix Noob V-Pred Eps v1.0 is an SDXL-based anime finetune of
noobai-XL-Vpred-1.0 that uses v-prediction parameterization for
text-to-image generation.

Available variants:
- GARAGE_MIX_NOOB_VPRED_EPS_V10_VPRED_SDXL: John6666/garage-mix-noob-vpred-eps-v10-vpred-sdxl text-to-image generation
"""

from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline

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
from ...stable_diffusion_xl.pytorch.src.model_utils import (
    stable_diffusion_preprocessing_xl,
)


REPO_ID = "John6666/garage-mix-noob-vpred-eps-v10-vpred-sdxl"


class ModelVariant(StrEnum):
    """Available Garage Mix Noob V-Pred Eps v1.0 V-Pred SDXL model variants."""

    GARAGE_MIX_NOOB_VPRED_EPS_V10_VPRED_SDXL = (
        "garage-mix-noob-vpred-eps-v10-vpred-sdxl"
    )


class ModelLoader(ForgeModel):
    """Garage Mix Noob V-Pred Eps v1.0 V-Pred SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.GARAGE_MIX_NOOB_VPRED_EPS_V10_VPRED_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.GARAGE_MIX_NOOB_VPRED_EPS_V10_VPRED_SDXL

    prompt = (
        "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="garage-mix-noob-vpred-eps-v10-vpred-sdxl",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the Garage Mix Noob V-Pred Eps v1.0 V-Pred SDXL pipeline.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Garage Mix Noob V-Pred Eps v1.0 V-Pred SDXL UNet model.

        Returns:
            dict: Keyword arguments for the UNet forward method.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(self.pipeline, self.prompt)

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
