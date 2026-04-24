# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-schnell GGUF model loader implementation for text-to-image generation.

This loader uses GGUF-quantized variants of the FLUX.1-schnell model from
lllyasviel/FLUX.1-schnell-gguf. The GGUF transformer is loaded via diffusers'
FluxTransformer2DModel.from_single_file with a local config to avoid fetching
from the gated black-forest-labs/FLUX.1-schnell repository.

Available variants:
- Q4_0: 4-bit quantization (default)
- Q8_0: 8-bit quantization
"""

import json
import os
import tempfile
from typing import Optional

import torch
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
from huggingface_hub import hf_hub_download

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

GGUF_REPO = "lllyasviel/FLUX.1-schnell-gguf"

# Config inferred from FLUX.1-schnell architecture (no guidance_embeds unlike dev).
# Provided to bypass the gated black-forest-labs/FLUX.1-schnell config lookup.
_TRANSFORMER_CONFIG = {
    "_class_name": "FluxTransformer2DModel",
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "guidance_embeds": False,
    "in_channels": 64,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "patch_size": 1,
    "pooled_projection_dim": 768,
}


class ModelVariant(StrEnum):
    """Available FLUX.1-schnell GGUF quantization variants."""

    Q4_0 = "Q4_0"
    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.Q4_0: "flux1-schnell-Q4_0.gguf",
    ModelVariant.Q8_0: "flux1-schnell-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """FLUX.1-schnell GGUF model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.Q4_0: ModelConfig(pretrained_model_name=GGUF_REPO),
        ModelVariant.Q8_0: ModelConfig(pretrained_model_name=GGUF_REPO),
    }

    DEFAULT_VARIANT = ModelVariant.Q4_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX.1-schnell GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGUF-quantized FLUX.1-schnell transformer.

        Returns:
            torch.nn.Module: The FLUX transformer model instance.
        """
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
        model_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)

        with tempfile.TemporaryDirectory() as config_dir:
            with open(os.path.join(config_dir, "config.json"), "w") as f:
                json.dump(_TRANSFORMER_CONFIG, f)

            self.transformer = FluxTransformer2DModel.from_single_file(
                model_path,
                config=config_dir,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
            )

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample inputs for the FLUX transformer.

        Returns:
            dict: Input tensors for the transformer model.
        """
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
