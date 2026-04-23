# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepBeepMeep/Flux single-file safetensors model loader implementation.

Loads FLUX.1 diffusion transformers from single-file bf16 safetensors
checkpoints hosted at DeepBeepMeep/Flux. The repository packages ungated
copies of the FLUX.1 family (dev, schnell, Krea-dev, SRPO-dev) as
single-file safetensors.

Reference: https://huggingface.co/DeepBeepMeep/Flux
"""

import json
import os
import tempfile
from typing import Optional

import torch
from diffusers import FluxTransformer2DModel
from huggingface_hub import hf_hub_download

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

SINGLE_FILE_REPO = "DeepBeepMeep/Flux"

# FLUX.1 transformer config (shared across dev/schnell/krea/srpo variants).
# Only guidance_embeds differs: True for the guidance-distilled dev family,
# False for the schnell timestep-distilled variant.
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
    """Available DeepBeepMeep/Flux single-file variants."""

    FLUX1_DEV_BF16 = "flux1-dev_bf16"
    FLUX1_SCHNELL_BF16 = "flux1-schnell_bf16"
    FLUX1_KREA_DEV_BF16 = "flux1-krea-dev_bf16"
    FLUX1_SRPO_DEV_BF16 = "flux1-srpo-dev_bf16"


_SINGLE_FILES = {
    ModelVariant.FLUX1_DEV_BF16: "flux1-dev_bf16.safetensors",
    ModelVariant.FLUX1_SCHNELL_BF16: "flux1-schnell_bf16.safetensors",
    ModelVariant.FLUX1_KREA_DEV_BF16: "flux1-krea-dev_bf16.safetensors",
    ModelVariant.FLUX1_SRPO_DEV_BF16: "flux1-srpo-dev_bf16.safetensors",
}

# Schnell is timestep-distilled and does not use the guidance embedding.
_SCHNELL_VARIANTS = {ModelVariant.FLUX1_SCHNELL_BF16}


class ModelLoader(ForgeModel):
    """DeepBeepMeep/Flux single-file safetensors model loader."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=SINGLE_FILE_REPO)
        for variant in ModelVariant
    }
    DEFAULT_VARIANT = ModelVariant.FLUX1_DEV_BF16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None
        self.guidance_scale = 0.0 if self._variant in _SCHNELL_VARIANTS else 3.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX_DEEPBEEPMEEP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _make_local_config_dir(self) -> str:
        """Create a temporary directory with the transformer config.json."""
        config = dict(_TRANSFORMER_CONFIG)
        config["guidance_embeds"] = self._variant not in _SCHNELL_VARIANTS

        config_dir = tempfile.mkdtemp()
        transformer_dir = os.path.join(config_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)
        with open(os.path.join(transformer_dir, "config.json"), "w") as f:
            json.dump(config, f)
        return config_dir

    def _load_transformer(
        self, dtype: torch.dtype = torch.bfloat16
    ) -> FluxTransformer2DModel:
        """Load the FLUX.1 transformer from a single-file safetensors checkpoint."""
        use_random_weights = os.environ.get("TT_RANDOM_WEIGHTS") or os.environ.get(
            "TT_COMPILE_ONLY_SYSTEM_DESC"
        )
        if use_random_weights:
            config = dict(_TRANSFORMER_CONFIG)
            config["guidance_embeds"] = self._variant not in _SCHNELL_VARIANTS
            self._transformer = FluxTransformer2DModel(**config)
            self._transformer = self._transformer.to(dtype=dtype)
            self._transformer.eval()
            return self._transformer

        model_path = hf_hub_download(
            repo_id=SINGLE_FILE_REPO,
            filename=_SINGLE_FILES[self._variant],
        )

        config_dir = self._make_local_config_dir()

        self._transformer = FluxTransformer2DModel.from_single_file(
            model_path,
            config=config_dir,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the FLUX.1 diffusion transformer."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, batch_size=1):
        """Generate sample inputs for the FLUX.1 transformer."""
        if self._transformer is None:
            self._load_transformer(
                dtype_override if dtype_override is not None else torch.bfloat16
            )

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        max_sequence_length = 256
        height = 128
        width = 128
        num_images_per_prompt = 1
        vae_scale_factor = 8

        num_channels_latents = self._transformer.config.in_channels // 4
        pooled_projection_dim = self._transformer.config.pooled_projection_dim
        joint_attention_dim = self._transformer.config.joint_attention_dim

        pooled_prompt_embeds = torch.randn(
            batch_size * num_images_per_prompt, pooled_projection_dim, dtype=dtype
        )
        prompt_embeds = torch.randn(
            batch_size * num_images_per_prompt,
            max_sequence_length,
            joint_attention_dim,
            dtype=dtype,
        )
        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        height_latent = 2 * (int(height) // (vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (vae_scale_factor * 2))

        latents = torch.randn(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent,
            width_latent,
            dtype=dtype,
        )
        latents = latents.view(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent // 2,
            2,
            width_latent // 2,
            2,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size * num_images_per_prompt,
            (height_latent // 2) * (width_latent // 2),
            num_channels_latents * 4,
        )

        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        if self.guidance_scale > 1.0:
            guidance = torch.full([batch_size], self.guidance_scale, dtype=dtype)
        else:
            guidance = None

        return {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
