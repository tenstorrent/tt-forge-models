# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-Krea-dev GGUF model loader implementation for text-to-image generation.

This loader uses GGUF-quantized variants of the FLUX.1-Krea-dev model from
InvokeAI/FLUX.1-Krea-dev-GGUF. The GGUF transformer is loaded via diffusers'
FluxTransformer2DModel.from_single_file with a local config to avoid gated
repo access.

Available variants:
- Q3_K_S: 3-bit quantization
- Q4_K_S: 4-bit quantization (default)
- Q4_K_M: 4-bit quantization (medium)
"""

import os
from typing import Optional

import torch
from diffusers import FluxTransformer2DModel
from diffusers.quantizers.quantization_config import GGUFQuantizationConfig

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

GGUF_REPO = "InvokeAI/FLUX.1-Krea-dev-GGUF"
GGUF_BASE_URL = f"https://huggingface.co/{GGUF_REPO}/blob/main"
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "transformer_config")


class ModelVariant(StrEnum):
    """Available FLUX.1-Krea-dev GGUF quantization variants."""

    Q3_K_S = "Q3_K_S"
    Q4_K_S = "Q4_K_S"
    Q4_K_M = "Q4_K_M"


_GGUF_FILES = {
    ModelVariant.Q3_K_S: "flux1-krea-dev-Q3_K_S.gguf",
    ModelVariant.Q4_K_S: "flux1-krea-dev-Q4_K_S.gguf",
    ModelVariant.Q4_K_M: "flux1-krea-dev-Q4_K_M.gguf",
}


class ModelLoader(ForgeModel):
    """FLUX.1-Krea-dev GGUF model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.Q3_K_S: ModelConfig(pretrained_model_name=GGUF_REPO),
        ModelVariant.Q4_K_S: ModelConfig(pretrained_model_name=GGUF_REPO),
        ModelVariant.Q4_K_M: ModelConfig(pretrained_model_name=GGUF_REPO),
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K_S

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None
        self.guidance_scale = 3.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX.1-Krea-dev GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        gguf_file = _GGUF_FILES[self._variant]
        gguf_url = f"{GGUF_BASE_URL}/{gguf_file}"

        load_kwargs = {}
        if dtype_override is not None:
            load_kwargs["torch_dtype"] = dtype_override

        self.transformer = FluxTransformer2DModel.from_single_file(
            gguf_url,
            config=_CONFIG_DIR,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype_override),
            **load_kwargs,
        )

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
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

        pooled_prompt_embeds = torch.randn(
            batch_size, config.pooled_projection_dim, dtype=dtype
        )

        text_ids = torch.zeros(max_sequence_length, 3).to(dtype=dtype)

        guidance = torch.full([batch_size], self.guidance_scale, dtype=dtype)

        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
