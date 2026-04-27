# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AlbedoBase XL (openart-custom/AlbedoBase) model loader implementation.

AlbedoBase XL is a Stable Diffusion XL text-to-image model by OpenArt.

Available variants:
- ALBEDO_BASE_XL: openart-custom/AlbedoBase text-to-image generation
"""

from typing import Optional

import torch
import torch.nn as nn

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
from .src.model_utils import load_pipe


class UNetWrapper(nn.Module):
    """Wraps UNet2DConditionModel to accept flattened added_cond_kwargs tensors.

    The TT XLA StableHLO backend cannot handle dict-typed arguments in the
    model forward pass. This wrapper accepts text_embeds and time_ids as
    separate tensor arguments and packs them into added_cond_kwargs before
    calling the underlying UNet.
    """

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )


REPO_ID = "openart-custom/AlbedoBase"


class ModelVariant(StrEnum):
    """Available AlbedoBase XL model variants."""

    ALBEDO_BASE_XL = "AlbedoBase_XL"


class ModelLoader(ForgeModel):
    """AlbedoBase XL model loader implementation."""

    _VARIANTS = {
        ModelVariant.ALBEDO_BASE_XL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.ALBEDO_BASE_XL

    # Shared configuration parameters
    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AlbedoBase XL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the AlbedoBase XL UNet model wrapped to accept flat tensors.

        Returns:
            torch.nn.Module: UNetWrapper around UNet2DConditionModel.
        """
        self.pipeline = load_pipe(self._variant_config.pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return UNetWrapper(self.pipeline.unet)

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the AlbedoBase XL model.

        Returns:
            dict: Keyword arguments for UNetWrapper.forward:
                - sample (torch.Tensor): Latent input for the UNet
                - timestep (torch.Tensor): Single timestep tensor
                - encoder_hidden_states (torch.Tensor): Encoded prompt embeddings
                - text_embeds (torch.Tensor): Pooled text embeddings (from added_cond_kwargs)
                - time_ids (torch.Tensor): Time IDs tensor (from added_cond_kwargs)
        """
        from ...stable_diffusion_xl.pytorch.src.model_utils import (
            stable_diffusion_preprocessing_xl,
        )

        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        # Convert alphas_cumprod to numpy to avoid numpy 2.0 DeprecationWarning
        # when scheduler.set_timesteps calls np.array() on torch tensors.
        import numpy as np
        if isinstance(self.pipeline.scheduler.alphas_cumprod, torch.Tensor):
            self.pipeline.scheduler.alphas_cumprod = (
                self.pipeline.scheduler.alphas_cumprod.numpy()
            )

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(self.pipeline, self.prompt)

        timestep = timesteps[0]
        text_embeds = added_cond_kwargs["text_embeds"]
        time_ids = added_cond_kwargs["time_ids"]

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)
            text_embeds = text_embeds.to(dtype_override)
            time_ids = time_ids.to(dtype_override)

        return {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }
