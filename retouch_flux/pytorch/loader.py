#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RetouchFLux LoRA model loader implementation.

Builds the FLUX.1-dev transformer architecture directly from the known public
config (no gated HuggingFace repo access required), applies the RetouchFLux
LoRA weights from the public TDN-M/RetouchFLux repo, and returns the
transformer for compilation.  All inputs are synthetic tensors derived from
the known FLUX.1-dev architecture dimensions.

Available variants:
- RETOUCH: RetouchFLux LoRA applied to FLUX.1-dev architecture transformer
"""

from typing import Optional

import torch
from diffusers.loaders.lora_pipeline import FluxLoraLoaderMixin
from diffusers.models import FluxTransformer2DModel

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

LORA_REPO = "TDN-M/RetouchFLux"
LORA_WEIGHT_NAME = "TDNM_Retouch.safetensors"

# Known public architecture spec for FLUX.1-dev transformer.
# Sourced from the (gated) black-forest-labs/FLUX.1-dev config.json.
_FLUX_DEV_CONFIG = dict(
    patch_size=1,
    in_channels=64,
    num_layers=19,
    num_single_layers=38,
    attention_head_dim=128,
    num_attention_heads=24,
    joint_attention_dim=4096,
    pooled_projection_dim=768,
    guidance_embeds=True,
    axes_dims_rope=[16, 56, 56],
)


class _LoRAApplicator(FluxLoraLoaderMixin):
    """Minimal mixin wrapper that lets us call load_lora_weights on a
    standalone FluxTransformer2DModel without a full pipeline object."""

    hf_device_map = None
    components = {}
    text_encoder = None
    text_encoder_name = "text_encoder"
    lora_scale = 1.0

    def __init__(self, transformer: FluxTransformer2DModel):
        self.transformer = transformer


class ModelVariant(StrEnum):
    """Available RetouchFLux LoRA variants."""

    RETOUCH = "Retouch"


class ModelLoader(ForgeModel):
    """RetouchFLux LoRA model loader."""

    _VARIANTS = {
        ModelVariant.RETOUCH: ModelConfig(
            pretrained_model_name=LORA_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.RETOUCH

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None
        self.guidance_scale = 3.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="RETOUCH_FLUX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_transformer(self, dtype: torch.dtype) -> FluxTransformer2DModel:
        transformer = FluxTransformer2DModel(**_FLUX_DEV_CONFIG).to(dtype)
        applicator = _LoRAApplicator(transformer)
        applicator.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHT_NAME)
        return transformer

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Build the FLUX.1-dev transformer and apply RetouchFLux LoRA weights.

        Returns:
            FluxTransformer2DModel with LoRA weights applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self.transformer is None:
            self.transformer = self._build_transformer(dtype)
        elif dtype_override is not None:
            self.transformer = self.transformer.to(dtype_override)
        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Generate synthetic inputs for FluxTransformer2DModel.forward.

        Dimensions are derived from the known FLUX.1-dev architecture and a
        128x128 latent resolution.

        Returns:
            dict with tensors matching FluxTransformer2DModel.forward signature.
        """
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        # Latent geometry for 128x128 image with vae_scale_factor=8
        height, width = 128, 128
        vae_scale_factor = 8
        height_latent = 2 * (height // (vae_scale_factor * 2))  # 16
        width_latent = 2 * (width // (vae_scale_factor * 2))  # 16
        h_packed = height_latent // 2  # 8
        w_packed = width_latent // 2  # 8
        num_channels_latents = _FLUX_DEV_CONFIG["in_channels"] // 4  # 16
        max_sequence_length = 256
        pooled_projection_dim = _FLUX_DEV_CONFIG["pooled_projection_dim"]  # 768
        joint_attention_dim = _FLUX_DEV_CONFIG["joint_attention_dim"]  # 4096

        latents = torch.randn(
            batch_size,
            h_packed * w_packed,
            num_channels_latents * 4,
            dtype=dtype,
        )

        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        return {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0], dtype=dtype).expand(batch_size),
            "guidance": torch.full([batch_size], self.guidance_scale, dtype=dtype),
            "pooled_projections": torch.randn(
                batch_size, pooled_projection_dim, dtype=dtype
            ),
            "encoder_hidden_states": torch.randn(
                batch_size, max_sequence_length, joint_attention_dim, dtype=dtype
            ),
            "txt_ids": torch.zeros(max_sequence_length, 3, dtype=dtype),
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
