# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wahtastic Furry Mix v9.2 Hotfix SDXL
(John6666/wahtastic-furry-mix-v92-hotfix-sdxl) model loader implementation.

Wahtastic Furry Mix v9.2 Hotfix is an SDXL-based anime/furry finetune of
noobai-XL-Vpred-1.0 that uses v-prediction parameterization for
text-to-image generation.

Available variants:
- WAHTASTIC_FURRY_MIX_V92_HOTFIX_SDXL: John6666/wahtastic-furry-mix-v92-hotfix-sdxl text-to-image generation
"""

from typing import Optional

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
from .src.model_utils import load_pipe, stable_diffusion_preprocessing_xl


REPO_ID = "John6666/wahtastic-furry-mix-v92-hotfix-sdxl"


class ModelVariant(StrEnum):
    """Available Wahtastic Furry Mix v9.2 Hotfix SDXL model variants."""

    WAHTASTIC_FURRY_MIX_V92_HOTFIX_SDXL = "wahtastic-furry-mix-v92-hotfix-sdxl"


class ModelLoader(ForgeModel):
    """Wahtastic Furry Mix v9.2 Hotfix SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.WAHTASTIC_FURRY_MIX_V92_HOTFIX_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAHTASTIC_FURRY_MIX_V92_HOTFIX_SDXL

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
            model="wahtastic-furry-mix-v92-hotfix-sdxl",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the UNet from the Wahtastic Furry Mix v9.2 Hotfix SDXL pipeline.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        self.pipeline = load_pipe(self._variant_config.pretrained_model_name)

        if dtype_override is not None:
            self.pipeline.unet = self.pipeline.unet.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Wahtastic Furry Mix v9.2 Hotfix SDXL UNet model.

        Returns:
            dict: Keyword arguments for the UNet forward method:
                - sample (torch.Tensor): Latent input for the UNet
                - timestep (torch.Tensor): Single timestep tensor
                - encoder_hidden_states (torch.Tensor): Encoded prompt embeddings
                - added_cond_kwargs (dict): Additional conditioning inputs
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
