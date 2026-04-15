# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hassaku XL Illustrious (John6666/hassaku-xl-illustrious-v21-sdxl) model loader implementation.

Hassaku XL Illustrious is an anime/illustration-focused SDXL fine-tune based on
OnomaAIResearch/Illustrious-xl-early-release-v0 for text-to-image generation.

Available variants:
- HASSAKU_XL_ILLUSTRIOUS_V21: John6666/hassaku-xl-illustrious-v21-sdxl text-to-image generation
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
from ...stable_diffusion_xl.pytorch.src.model_utils import (
    load_pipe,
    stable_diffusion_preprocessing_xl,
)


class ModelVariant(StrEnum):
    """Available Hassaku XL Illustrious model variants."""

    HASSAKU_XL_ILLUSTRIOUS_V21 = "hassaku-xl-illustrious-v21-sdxl"


class ModelLoader(ForgeModel):
    """Hassaku XL Illustrious model loader implementation."""

    _VARIANTS = {
        ModelVariant.HASSAKU_XL_ILLUSTRIOUS_V21: ModelConfig(
            pretrained_model_name="John6666/hassaku-xl-illustrious-v21-sdxl",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.HASSAKU_XL_ILLUSTRIOUS_V21

    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Hassaku XL Illustrious",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the Hassaku XL Illustrious pipeline.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.pipeline = load_pipe(pretrained_model_name)

        # Hassaku model ships with float16 weights; ensure all components are
        # float32 so that preprocessing (encode_prompt) does not hit dtype
        # mismatches on CPU.
        self.pipeline.to(torch.float32)

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Hassaku XL Illustrious UNet model.

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
