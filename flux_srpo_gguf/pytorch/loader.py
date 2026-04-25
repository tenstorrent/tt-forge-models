# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX-SRPO GGUF model loader implementation for text-to-image generation.

This loader uses GGUF-quantized variants of the FLUX-SRPO model published
in MonsterMMORPG/Wan_GGUF. FLUX-SRPO is a Stepwise Relative Preference
Optimization fine-tune of FLUX.1-dev. The GGUF transformer is loaded via
diffusers' FluxTransformer2DModel.from_single_file with a locally-materialised
config to avoid the gated black-forest-labs/FLUX.1-dev and FLUX.1-Depth-dev
repos. Synthetic inputs are generated from the transformer config.
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

GGUF_REPO = "MonsterMMORPG/Wan_GGUF"

# Standard FLUX.1-dev transformer architecture; FLUX-SRPO is a dev fine-tune
# so it uses guidance_embeds=True (classifier-free guidance distillation).
_TRANSFORMER_CONFIG = {
    "_class_name": "FluxTransformer2DModel",
    "_diffusers_version": "0.37.1",
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "guidance_embeds": True,
    "in_channels": 64,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "patch_size": 1,
    "pooled_projection_dim": 768,
}


class ModelVariant(StrEnum):
    """Available FLUX-SRPO GGUF quantization variants."""

    Q4_K = "Q4_K"
    Q5_K = "Q5_K"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.Q4_K: "FLUX-SRPO-GGUF_Q4_K.gguf",
    ModelVariant.Q5_K: "FLUX-SRPO-GGUF_Q5_K.gguf",
    ModelVariant.Q6_K: "FLUX-SRPO-GGUF_Q6_K.gguf",
    ModelVariant.Q8_0: "FLUX-SRPO-GGUF_Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """FLUX-SRPO GGUF model loader for text-to-image generation."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=GGUF_REPO) for variant in _GGUF_FILES
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX-SRPO GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _make_local_config_dir(self):
        """Materialise a transformer/config.json so from_single_file can locate it."""
        config_dir = tempfile.mkdtemp()
        transformer_dir = os.path.join(config_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)
        with open(os.path.join(transformer_dir, "config.json"), "w") as f:
            json.dump(_TRANSFORMER_CONFIG, f)
        return config_dir

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16):
        """Load the GGUF-quantized FluxTransformer2DModel."""
        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
        config_dir = self._make_local_config_dir()

        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)
        self.transformer = FluxTransformer2DModel.from_single_file(
            gguf_path,
            config=config_dir,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )

        return self.transformer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GGUF-quantized FLUX-SRPO transformer.

        Returns:
            torch.nn.Module: The FLUX-SRPO transformer model instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self.transformer is None:
            self._load_pipeline(dtype)
        elif dtype_override is not None:
            self.transformer = self.transformer.to(dtype=dtype_override)
        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare synthetic inputs for the FLUX-SRPO transformer.

        Returns:
            dict: Input tensors for the transformer model.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.transformer is None:
            self._load_pipeline(dtype)

        config = self.transformer.config
        max_sequence_length = 256
        height = 128
        width = 128
        vae_scale_factor = 8
        num_images_per_prompt = 1

        # Latent dimensions
        height_latent = 2 * (int(height) // (vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (vae_scale_factor * 2))
        seq_len = (height_latent // 2) * (width_latent // 2)

        # FLUX packs 2x2 spatial patches, so the channel dim is
        # num_channels_latents * 4, matching the transformer's in_channels.
        num_channels_latents = config.in_channels // 4
        latents = torch.randn(
            batch_size * num_images_per_prompt,
            seq_len,
            num_channels_latents * 4,
            dtype=dtype,
        )

        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        # Synthetic text embeddings (no text encoders needed for compile-only testing)
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
        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        # FLUX.1-dev uses classifier-free guidance distillation
        guidance = torch.tensor([3.5], dtype=dtype).expand(
            batch_size * num_images_per_prompt
        )

        return {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0], dtype=dtype).expand(
                batch_size * num_images_per_prompt
            ),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
