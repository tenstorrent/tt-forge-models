# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2-klein-4B GGUF (leejet/FLUX.2-klein-4B-GGUF) model loader implementation.

FLUX.2-klein is the 4B parameter distilled text-to-image variant of FLUX.2 from
Black Forest Labs. This loader consumes the GGUF-quantized checkpoints published
by leejet for stable-diffusion.cpp and loads them via diffusers'
Flux2Transformer2DModel.from_single_file.

Available variants:
- FLUX2_KLEIN_4B_Q4_0: Q4_0 quantized variant (~2.46 GB)
- FLUX2_KLEIN_4B_Q8_0: Q8_0 quantized variant (~4.3 GB)
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
from .src.model_utils import load_flux2_klein_gguf_transformer

REPO_ID = "leejet/FLUX.2-klein-4B-GGUF"


class ModelVariant(StrEnum):
    """Available FLUX.2-klein-4B GGUF model variants."""

    FLUX2_KLEIN_4B_Q4_0 = "flux2_klein_4b_Q4_0"
    FLUX2_KLEIN_4B_Q8_0 = "flux2_klein_4b_Q8_0"


class ModelLoader(ForgeModel):
    """FLUX.2-klein-4B GGUF model loader implementation."""

    _VARIANTS = {
        ModelVariant.FLUX2_KLEIN_4B_Q4_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.FLUX2_KLEIN_4B_Q8_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FLUX2_KLEIN_4B_Q4_0

    GGUF_FILES = {
        ModelVariant.FLUX2_KLEIN_4B_Q4_0: "flux-2-klein-4b-Q4_0.gguf",
        ModelVariant.FLUX2_KLEIN_4B_Q8_0: "flux-2-klein-4b-Q8_0.gguf",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX.2-klein-4B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FLUX.2-klein transformer from GGUF checkpoint.

        Returns:
            torch.nn.Module: The FLUX.2-klein transformer model instance.
        """
        if self.transformer is None:
            gguf_file = self.GGUF_FILES[self._variant]
            self.transformer = load_flux2_klein_gguf_transformer(REPO_ID, gguf_file)

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model.

        Returns:
            dict: Input tensors for the FLUX.2-klein transformer model.
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

        # Timestep
        timestep = torch.tensor([1.0 / 1000], dtype=dtype).expand(batch_size)

        inputs = {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_ids,
            "joint_attention_kwargs": {},
        }

        return inputs
