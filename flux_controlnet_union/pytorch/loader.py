# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX ControlNet Union model loader implementation
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
from .src.model_utils import load_flux_transformer


class ModelVariant(StrEnum):
    """Available FLUX ControlNet Union model variants."""

    FLUX_1_DEV_CONTROLNET_UNION = "FLUX.1-dev-Controlnet-Union"


class ModelLoader(ForgeModel):
    """FLUX ControlNet Union model loader implementation."""

    _VARIANTS = {
        ModelVariant.FLUX_1_DEV_CONTROLNET_UNION: ModelConfig(
            pretrained_model_name="InstantX/FLUX.1-dev-Controlnet-Union",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FLUX_1_DEV_CONTROLNET_UNION

    base_model = "black-forest-labs/FLUX.1-dev"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX ControlNet Union",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FLUX transformer model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The FLUX transformer model instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.transformer = load_flux_transformer(
            self._variant_config.pretrained_model_name,
            self.base_model,
            dtype=dtype,
        )
        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the FLUX ControlNet Union model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the transformer model.
        """
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        max_sequence_length = 256
        guidance_scale = 3.5
        height = 128
        width = 128
        num_images_per_prompt = 1
        vae_scale_factor = 8
        num_channels_latents = config.in_channels // 4

        height_latent = 2 * (int(height) // (vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        latents = torch.randn(
            batch_size * num_images_per_prompt,
            h_packed * w_packed,
            num_channels_latents * 4,
            dtype=dtype,
        )

        pooled_prompt_embeds = torch.randn(
            batch_size * num_images_per_prompt,
            config.pooled_projection_dim,
            dtype=dtype,
        )
        prompt_embeds = torch.randn(
            batch_size * num_images_per_prompt,
            max_sequence_length,
            config.joint_attention_dim,
            dtype=dtype,
        )

        text_ids = torch.zeros(prompt_embeds.shape[1], 3, dtype=dtype)

        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        guidance = torch.full([batch_size], guidance_scale, dtype=dtype)

        inputs = {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }

        return inputs
