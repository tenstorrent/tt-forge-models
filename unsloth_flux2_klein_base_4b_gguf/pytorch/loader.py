# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unsloth FLUX.2-klein-base-4B GGUF model loader implementation for text-to-image generation.

This loader wraps GGUF-quantized variants of FLUX.2-klein-base-4B published by Unsloth at
unsloth/FLUX.2-klein-base-4B-GGUF. The GGUF transformer is loaded via diffusers'
Flux2Transformer2DModel.from_single_file.
"""

import json
import os
import tempfile
from typing import Optional

import torch
from diffusers import GGUFQuantizationConfig
from diffusers.models import Flux2Transformer2DModel
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

GGUF_REPO = "unsloth/FLUX.2-klein-base-4B-GGUF"


class ModelVariant(StrEnum):
    """Available Unsloth FLUX.2-klein-base-4B GGUF quantization variants."""

    BF16 = "BF16"
    F16 = "F16"
    Q2_K = "Q2_K"
    Q3_K_M = "Q3_K_M"
    Q3_K_S = "Q3_K_S"
    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q4_K_M = "Q4_K_M"
    Q4_K_S = "Q4_K_S"
    Q5_0 = "Q5_0"
    Q5_1 = "Q5_1"
    Q5_K_M = "Q5_K_M"
    Q5_K_S = "Q5_K_S"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"


_GGUF_FILES = {
    variant: f"flux-2-klein-base-4b-{variant.value}.gguf" for variant in ModelVariant
}


class ModelLoader(ForgeModel):
    """Unsloth FLUX.2-klein-base-4B GGUF model loader for text-to-image generation."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=GGUF_REPO) for variant in _GGUF_FILES
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None
        self.guidance_scale = 4.0

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Unsloth FLUX.2-klein-base-4B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGUF-quantized FLUX.2-klein-base-4B transformer.

        Returns:
            torch.nn.Module: The FLUX.2-klein-base-4B transformer model instance.
        """
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        model_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)

        # FLUX.2-klein-base-4B config inferred from GGUF tensor shapes.
        # black-forest-labs/FLUX.2-dev (the default diffusers falls back to) is a
        # gated repo; supplying a local config avoids that network access.
        config_dict = {
            "_class_name": "Flux2Transformer2DModel",
            "attention_head_dim": 128,
            "axes_dims_rope": [32, 32, 32, 32],
            "guidance_embeds": False,
            "in_channels": 128,
            "joint_attention_dim": 7680,
            "mlp_ratio": 3.0,
            "num_attention_heads": 24,
            "num_layers": 5,
            "num_single_layers": 20,
            "out_channels": None,
            "patch_size": 1,
            "rope_theta": 2000,
            "timestep_guidance_channels": 256,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config_dict, f)
            self.transformer = Flux2Transformer2DModel.from_single_file(
                model_path,
                config=tmpdir,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
            )

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the FLUX.2-klein-base-4B transformer.

        Returns:
            dict: Input tensors for the Flux2Transformer2DModel.
        """
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self.transformer.config

        # Image dimensions
        height = 128
        width = 128
        vae_scale_factor = 8
        num_channels_latents = config.in_channels // 4

        # Prepare latents: VAE compresses by vae_scale_factor, then pack 2x2 patches
        height_latent = 2 * (height // (vae_scale_factor * 2))
        width_latent = 2 * (width // (vae_scale_factor * 2))
        h_packed = height_latent // 2
        w_packed = width_latent // 2

        # Create latent tensor (B, C, H, W) then pack to (B, H*W, C)
        latents = torch.randn(
            batch_size, num_channels_latents * 4, h_packed, w_packed, dtype=dtype
        )

        # Prepare latent image IDs (B, H*W, 4)
        t = torch.arange(1)
        h = torch.arange(h_packed)
        w = torch.arange(w_packed)
        l = torch.arange(1)
        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=dtype)

        # Pack latents: (B, C, H, W) -> (B, H*W, C)
        latents = latents.reshape(batch_size, num_channels_latents * 4, -1).permute(
            0, 2, 1
        )

        # Prompt embeddings: use random tensors matching joint_attention_dim
        max_sequence_length = 256
        joint_attention_dim = config.joint_attention_dim
        prompt_embeds = torch.randn(
            batch_size, max_sequence_length, joint_attention_dim, dtype=dtype
        )

        # Text IDs (B, seq_len, 4)
        t = torch.arange(1)
        h = torch.arange(1)
        w = torch.arange(1)
        l = torch.arange(max_sequence_length)
        text_ids = torch.cartesian_prod(t, h, w, l)
        text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=dtype)

        # Guidance
        guidance = torch.full([batch_size], self.guidance_scale, dtype=dtype)

        # Timestep
        timestep = torch.tensor([1.0 / 1000], dtype=dtype).expand(batch_size)

        inputs = {
            "hidden_states": latents,
            "timestep": timestep,
            "guidance": guidance,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_ids,
            "joint_attention_kwargs": {},
        }

        return inputs
