# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Amanatsu Illustrious v1.1 SDXL (John6666/amanatsu-illustrious-v11-sdxl) model loader implementation.

Amanatsu Illustrious is an anime-focused Stable Diffusion XL fine-tune
optimized for detailed anime illustration generation.

Available variants:
- AMANATSU_ILLUSTRIOUS_V11: John6666/amanatsu-illustrious-v11-sdxl text-to-image generation
"""

from typing import Optional

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
from .src.model_utils import load_pipe, stable_diffusion_preprocessing_xl


REPO_ID = "John6666/amanatsu-illustrious-v11-sdxl"


class ModelVariant(StrEnum):
    """Available Amanatsu Illustrious model variants."""

    AMANATSU_ILLUSTRIOUS_V11 = "amanatsu-illustrious-v11"


class ModelLoader(ForgeModel):
    """Amanatsu Illustrious SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.AMANATSU_ILLUSTRIOUS_V11: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.AMANATSU_ILLUSTRIOUS_V11

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
            model="Amanatsu Illustrious",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the Amanatsu Illustrious SDXL pipeline.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.pipeline = load_pipe(
            self._variant_config.pretrained_model_name, dtype=dtype
        )
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Amanatsu Illustrious UNet model.

        Uses 512x512 resolution to keep the latent size (64x64) tractable for
        CPU reference runs.

        Returns:
            dict: Keyword arguments for the UNet forward method:
                - sample (torch.Tensor): Latent input for the UNet
                - timestep (torch.Tensor): Single timestep tensor
                - encoder_hidden_states (torch.Tensor): Encoded prompt embeddings
                - added_cond_kwargs (dict): Additional conditioning inputs
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self.pipeline is None:
            self.load_model(dtype_override=dtype)

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(
            self.pipeline, self.prompt, height=512, width=512
        )

        timestep = timesteps[0]

        return {
            "sample": latent_model_input.to(dtype),
            "timestep": timestep.to(dtype),
            "encoder_hidden_states": prompt_embeds.to(dtype),
            "added_cond_kwargs": added_cond_kwargs,
        }
