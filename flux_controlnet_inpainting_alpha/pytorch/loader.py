# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX ControlNet Inpainting Alpha model loader implementation.

Loads the alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha controlnet
directly (no gated FLUX.1-dev base model required). The checkpoint has
extra_condition_channels=4 which the current diffusers FluxControlNetModel does
not yet support; we load with ignore_mismatched_sizes=True for compile-only use.
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
from .src.model_utils import load_flux_controlnet_inpainting_alpha


class ModelVariant(StrEnum):
    """Available FLUX ControlNet Inpainting Alpha model variants."""

    FLUX_1_DEV_CONTROLNET_INPAINTING_ALPHA = "FLUX.1-dev-Controlnet-Inpainting-Alpha"


class ModelLoader(ForgeModel):
    """FLUX ControlNet Inpainting Alpha model loader implementation."""

    _VARIANTS = {
        ModelVariant.FLUX_1_DEV_CONTROLNET_INPAINTING_ALPHA: ModelConfig(
            pretrained_model_name="alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FLUX_1_DEV_CONTROLNET_INPAINTING_ALPHA

    guidance_scale = 3.5

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX ControlNet Inpainting Alpha",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FLUX ControlNet Inpainting Alpha model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The FluxControlNetModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.controlnet = load_flux_controlnet_inpainting_alpha(
            self._variant_config.pretrained_model_name, dtype=dtype
        )
        return self.controlnet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Generate sample inputs for the FLUX ControlNet Inpainting Alpha model.

        Args:
            dtype_override: Optional torch.dtype to override the default input dtype.
            batch_size: Batch size (default: 1).

        Returns:
            dict: Input tensors matching FluxControlNetModel.forward.
        """
        if self.controlnet is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.controlnet.config

        height = 128
        width = 128
        vae_scale_factor = 8
        max_sequence_length = 256
        num_channels_latents = config.in_channels // 4

        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        latents = torch.randn(
            batch_size,
            h_packed * w_packed,
            num_channels_latents * 4,
            dtype=dtype,
        )

        # controlnet_cond: inpainting condition in packed latent format.
        # With ignore_mismatched_sizes=True the controlnet_x_embedder has
        # in_channels=64 (matching latents), so controlnet_cond uses same shape.
        controlnet_cond = torch.randn_like(latents)

        pooled_prompt_embeds = torch.randn(
            batch_size, config.pooled_projection_dim, dtype=dtype
        )
        prompt_embeds = torch.randn(
            batch_size,
            max_sequence_length,
            config.joint_attention_dim,
            dtype=dtype,
        )

        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        guidance = (
            torch.full([batch_size], self.guidance_scale, dtype=dtype)
            if config.guidance_embeds
            else None
        )

        inputs = {
            "hidden_states": latents,
            "controlnet_cond": controlnet_cond,
            "controlnet_mode": None,
            "conditioning_scale": 1.0,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "img_ids": latent_image_ids,
            "txt_ids": text_ids,
            "guidance": guidance,
            "joint_attention_kwargs": {},
        }

        return inputs
