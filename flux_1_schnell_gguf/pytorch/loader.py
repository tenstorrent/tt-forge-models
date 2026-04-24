# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-schnell GGUF model loader implementation for text-to-image generation.

This loader uses GGUF-quantized variants of the FLUX.1-schnell model from
lllyasviel/FLUX.1-schnell-gguf. The GGUF transformer is loaded via diffusers'
FluxTransformer2DModel.from_single_file using a bundled local config (avoiding
the gated black-forest-labs/FLUX.1-schnell base repository). Sample inputs are
synthesized directly from the known transformer dimensions.

Available variants:
- Q4_0: 4-bit quantization (default)
- Q8_0: 8-bit quantization
"""

import os
from typing import Optional

import torch
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig

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

# Local FluxTransformer2DModel config for FLUX.1-schnell (in_channels=64 for standard
# text-to-image, guidance_embeds=False for guidance-distilled model).
_TRANSFORMER_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "transformer_config")

# FLUX VAE scale factor (spatial downsampling ratio)
_VAE_SCALE_FACTOR = 16


class ModelVariant(StrEnum):
    """Available FLUX.1-schnell GGUF quantization variants."""

    Q4_0 = "Q4_0"
    Q8_0 = "Q8_0"


# Map variants to their GGUF filenames
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
        self._transformer = None

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

    def _load_transformer(self, dtype: torch.dtype = torch.bfloat16):
        """Load the GGUF-quantized FluxTransformer2DModel using a local config."""
        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
        return FluxTransformer2DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/blob/main/{gguf_file}",
            config=_TRANSFORMER_CONFIG_DIR,
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGUF-quantized FLUX transformer.

        Returns:
            torch.nn.Module: The FLUX transformer model instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            self._transformer = self._load_transformer(dtype)
        elif dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample inputs for the FLUX transformer.

        Inputs are synthesized directly from the known FLUX.1-schnell transformer
        dimensions without requiring the gated black-forest-labs/FLUX.1-schnell
        base pipeline.

        Returns:
            dict: Input tensors for the transformer model.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        max_sequence_length = 256  # T5-XXL sequence length
        height = 128
        width = 128
        num_images_per_prompt = 1

        # Latent dimensions (FLUX uses vae_scale_factor=16 with 2x packing)
        height_latent = 2 * (height // (_VAE_SCALE_FACTOR * 2))
        width_latent = 2 * (width // (_VAE_SCALE_FACTOR * 2))
        seq_len = (height_latent // 2) * (width_latent // 2)

        # Packed latents: 64 channels (16 VAE channels * 4 for 2x2 spatial packing)
        hidden_states = torch.zeros(
            batch_size * num_images_per_prompt, seq_len, 64, dtype=dtype
        )

        # CLIP-L pooled text embedding (pooled_projection_dim=768)
        pooled_prompt_embeds = torch.zeros(
            batch_size * num_images_per_prompt, 768, dtype=dtype
        )

        # T5-XXL text embedding (joint_attention_dim=4096)
        prompt_embeds = torch.zeros(
            batch_size * num_images_per_prompt, max_sequence_length, 4096, dtype=dtype
        )

        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        # Latent image IDs
        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": None,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
