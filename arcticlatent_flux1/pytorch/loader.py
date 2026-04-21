# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ArcticLatent FLUX.1 model loader implementation for text-to-image generation.

Loads FLUX.1 transformer weights from the ungated arcticlatent/flux1 bundle
repository, which redistributes flux1-dev, flux1-schnell and flux1-kontext-dev
unet safetensors files. Since the repo only ships the raw transformer weights
(no config.json), a local transformer config is materialised at load time.

Repository:
- https://huggingface.co/arcticlatent/flux1
"""
import json
import os
import tempfile

import torch
from diffusers.models import FluxTransformer2DModel
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

REPO_ID = "arcticlatent/flux1"
_REPO_BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"

# Standard FLUX.1 transformer architecture (shared by Dev, Schnell and Kontext).
_TRANSFORMER_CONFIG = {
    "_class_name": "FluxTransformer2DModel",
    "_diffusers_version": "0.37.1",
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "in_channels": 64,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "patch_size": 1,
    "pooled_projection_dim": 768,
}


class ModelVariant(StrEnum):
    """Available ArcticLatent FLUX.1 variants."""

    DEV_FP16 = "Dev_FP16"
    SCHNELL_FP16 = "Schnell_FP16"
    KONTEXT_FP16 = "Kontext_FP16"


class ModelLoader(ForgeModel):
    """ArcticLatent FLUX.1 loader for text-to-image generation using single-file safetensors."""

    _VARIANTS = {
        ModelVariant.DEV_FP16: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.SCHNELL_FP16: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.KONTEXT_FP16: ModelConfig(pretrained_model_name=REPO_ID),
    }

    _SAFETENSOR_PATHS = {
        ModelVariant.DEV_FP16: "unet/dev/flux1-dev-fp16.safetensors",
        ModelVariant.SCHNELL_FP16: "unet/schnell/flux1-schnell-fp16.safetensors",
        ModelVariant.KONTEXT_FP16: "unet/kontext/flux1-kontext-dev-fp16.safetensors",
    }

    # Schnell is distilled and does not use guidance embeddings.
    _GUIDANCE_EMBEDS = {
        ModelVariant.DEV_FP16: True,
        ModelVariant.SCHNELL_FP16: False,
        ModelVariant.KONTEXT_FP16: True,
    }

    DEFAULT_VARIANT = ModelVariant.DEV_FP16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None
        self.guidance_scale = 3.5 if self._GUIDANCE_EMBEDS[self._variant] else 0.0

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ArcticLatent-FLUX.1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _make_local_config_dir(self):
        """Materialize a transformer/config.json so from_single_file can locate it."""
        config = dict(_TRANSFORMER_CONFIG)
        config["guidance_embeds"] = self._GUIDANCE_EMBEDS[self._variant]

        config_dir = tempfile.mkdtemp()
        transformer_dir = os.path.join(config_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)
        with open(os.path.join(transformer_dir, "config.json"), "w") as f:
            json.dump(config, f)
        return config_dir

    def _load_transformer(self, dtype_override=None):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config_dir = self._make_local_config_dir()

        safetensor_url = f"{_REPO_BASE_URL}/{self._SAFETENSOR_PATHS[self._variant]}"

        self._transformer = FluxTransformer2DModel.from_single_file(
            safetensor_url,
            config=config_dir,
            subfolder="transformer",
            torch_dtype=dtype,
        )

        return self._transformer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FLUX.1 transformer for the selected variant."""
        if self._transformer is None:
            self._load_transformer(dtype_override=dtype_override)

        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype_override)

        return self._transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Generate synthetic inputs for the FLUX.1 transformer forward pass."""
        if self._transformer is None:
            self._load_transformer(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self._transformer.config

        # Small spatial dimensions for compile-only testing.
        height = 128
        width = 128
        vae_scale_factor = 8
        num_channels_latents = config.in_channels // 4

        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        # Pack latents to (B, H*W, C).
        latents = torch.randn(
            batch_size, num_channels_latents * 4, h_packed, w_packed, dtype=dtype
        )
        latents = latents.reshape(batch_size, num_channels_latents * 4, -1).permute(
            0, 2, 1
        )

        # Latent image IDs.
        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        # Text embeddings.
        max_sequence_length = 256
        prompt_embeds = torch.randn(
            batch_size, max_sequence_length, config.joint_attention_dim, dtype=dtype
        )
        pooled_prompt_embeds = torch.randn(
            batch_size, config.pooled_projection_dim, dtype=dtype
        )
        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        if self._GUIDANCE_EMBEDS[self._variant]:
            guidance = torch.full([batch_size], self.guidance_scale, dtype=dtype)
        else:
            guidance = None

        inputs = {
            "hidden_states": latents,
            "timestep": timestep,
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }

        return inputs
