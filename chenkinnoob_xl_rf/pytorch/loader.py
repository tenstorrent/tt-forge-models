# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ChenkinNoob-XL Rectified Flow (ChenkinRF/ChenkinNoob-XL-v0.3-Rectified-Flow) model loader.

ChenkinNoob-XL is a Rectified Flow variant of Stable Diffusion XL, fine-tuned on
Danbooru for anime/illustration-style text-to-image generation. It uses SD3-style
flow-matching sampling rather than standard epsilon/v-prediction scheduling.

Available variants:
- V0_3: ChenkinNoob-XL v0.3 Rectified Flow
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

REPO_ID = "ChenkinRF/ChenkinNoob-XL-v0.3-Rectified-Flow"
CHECKPOINT_FILE = "ChenkinNoob-XL-v0.3-Rectified-Flow.safetensors"


class ModelVariant(StrEnum):
    """Available ChenkinNoob-XL Rectified Flow model variants."""

    V0_3 = "v0.3"


class ModelLoader(ForgeModel):
    """ChenkinNoob-XL Rectified Flow model loader implementation."""

    _VARIANTS = {
        ModelVariant.V0_3: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.V0_3

    prompt = "masterpiece, best quality, aesthetic, 1girl, solo, looking at viewer"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ChenkinNoob-XL Rectified Flow",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the ChenkinNoob-XL pipeline.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        self.pipeline = load_pipe(REPO_ID, CHECKPOINT_FILE)

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the UNet model.

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
