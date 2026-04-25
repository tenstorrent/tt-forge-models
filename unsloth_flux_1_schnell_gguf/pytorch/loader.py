# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unsloth FLUX.1-schnell GGUF model loader implementation for text-to-image generation.

This loader uses GGUF-quantized variants of the FLUX.1-schnell model from
unsloth/FLUX.1-schnell-GGUF. The GGUF transformer is loaded via diffusers'
FluxTransformer2DModel.from_single_file with a local config to avoid needing
access to the gated black-forest-labs/FLUX.1-schnell repository.

Available variants:
- Q4_K_M: 4-bit quantization (default)
- Q8_0: 8-bit quantization
"""

from pathlib import Path
from typing import Optional

import torch
from diffusers import GGUFQuantizationConfig
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

GGUF_REPO = "unsloth/FLUX.1-schnell-GGUF"


class ModelVariant(StrEnum):
    """Available Unsloth FLUX.1-schnell GGUF quantization variants."""

    Q4_K_M = "Q4_K_M"
    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.Q4_K_M: "flux1-schnell-Q4_K_M.gguf",
    ModelVariant.Q8_0: "flux1-schnell-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """Unsloth FLUX.1-schnell GGUF model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.Q4_K_M: ModelConfig(pretrained_model_name=GGUF_REPO),
        ModelVariant.Q8_0: ModelConfig(pretrained_model_name=GGUF_REPO),
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Unsloth FLUX.1-schnell GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGUF-quantized FLUX.1-schnell transformer."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
        gguf_file = _GGUF_FILES[self._variant]
        config_dir = str(Path(__file__).parent / "transformer_config")
        self.transformer = FluxTransformer2DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/blob/main/{gguf_file}",
            config=config_dir,
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample inputs for the FLUX.1-schnell transformer."""
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        height = 128
        width = 128
        vae_scale_factor = 8
        num_channels_latents = config.in_channels // 4

        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        latents = torch.randn(
            batch_size, num_channels_latents * 4, h_packed, w_packed, dtype=dtype
        )
        latents = latents.reshape(batch_size, num_channels_latents * 4, -1).permute(
            0, 2, 1
        )

        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        max_sequence_length = 256
        joint_attention_dim = config.joint_attention_dim
        prompt_embeds = torch.randn(
            batch_size, max_sequence_length, joint_attention_dim, dtype=dtype
        )

        pooled_projection_dim = config.pooled_projection_dim
        pooled_prompt_embeds = torch.randn(
            batch_size, pooled_projection_dim, dtype=dtype
        )

        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)
        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "guidance": None,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
