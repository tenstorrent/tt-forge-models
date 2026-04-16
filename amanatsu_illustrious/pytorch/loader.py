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

import os
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

# SDXL UNet input shapes (with classifier-free guidance, batch doubles)
_SDXL_LATENT_CHANNELS = 4
_SDXL_LATENT_HEIGHT = 64  # 512 / vae_scale_factor(8)
_SDXL_LATENT_WIDTH = 64
_SDXL_SEQ_LEN = 77
_SDXL_ENCODER_HIDDEN_SIZE = 2048  # SDXL dual text encoder concat output
_SDXL_POOLED_SIZE = 1280  # SDXL pooled text embed size
_SDXL_TIME_IDS = 6  # SDXL added_cond_kwargs time_ids length


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

        When TT_RANDOM_WEIGHTS=1, loads only the UNet config and uses random
        weights to avoid downloading the full 6.5GB model on compile-only systems.

        Returns:
            torch.nn.Module: The UNet model used for denoising.
        """
        from diffusers import UNet2DConditionModel

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if os.environ.get("TT_RANDOM_WEIGHTS", "") == "1":
            # Download only the tiny UNet config JSON, then random-init
            unet_config = UNet2DConditionModel.load_config(
                self._variant_config.pretrained_model_name, subfolder="unet"
            )
            unet = UNet2DConditionModel.from_config(unet_config)
            unet = unet.to(dtype).eval()
            return unet

        self.pipeline = load_pipe(
            self._variant_config.pretrained_model_name, dtype=dtype
        )
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Amanatsu Illustrious UNet model.

        Uses 512x512 resolution to keep the latent size (64x64) tractable for
        CPU reference runs. When TT_RANDOM_WEIGHTS=1, uses random tensors with
        the correct SDXL input shapes.

        Returns:
            dict: Keyword arguments for the UNet forward method:
                - sample (torch.Tensor): Latent input for the UNet
                - timestep (torch.Tensor): Single timestep tensor
                - encoder_hidden_states (torch.Tensor): Encoded prompt embeddings
                - added_cond_kwargs (dict): Additional conditioning inputs
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if os.environ.get("TT_RANDOM_WEIGHTS", "") == "1":
            # CFG doubles the batch dimension
            cfg_batch = batch_size * 2
            torch.manual_seed(42)
            return {
                "sample": torch.randn(
                    cfg_batch,
                    _SDXL_LATENT_CHANNELS,
                    _SDXL_LATENT_HEIGHT,
                    _SDXL_LATENT_WIDTH,
                    dtype=dtype,
                ),
                "timestep": torch.tensor(999, dtype=dtype),
                "encoder_hidden_states": torch.randn(
                    cfg_batch, _SDXL_SEQ_LEN, _SDXL_ENCODER_HIDDEN_SIZE, dtype=dtype
                ),
                "added_cond_kwargs": {
                    "text_embeds": torch.randn(
                        cfg_batch, _SDXL_POOLED_SIZE, dtype=dtype
                    ),
                    "time_ids": torch.zeros(cfg_batch, _SDXL_TIME_IDS, dtype=dtype),
                },
            }

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
