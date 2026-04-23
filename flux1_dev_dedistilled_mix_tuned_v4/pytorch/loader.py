# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Flux1-Dev-DedistilledMixTuned-v4 model loader implementation for text-to-image generation.

Loads the FLUX.1-dev-based transformer from a single fp8 safetensors checkpoint
hosted at wikeeyang/Flux1-Dev-DedistilledMixTuned-v4. Uses a local transformer
config and synthetic inputs to avoid requiring the full gated pipeline.
"""
import json
import os
import tempfile
from typing import Optional

import torch
from diffusers.models import FluxTransformer2DModel
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

_REPO_ID = "wikeeyang/Flux1-Dev-DedistilledMixTuned-v4"
_FILENAME = "Flux1-Dev-DedistilledMixTuned-V4-fp8.safetensors"

# Standard FLUX.1-dev transformer architecture config.
# guidance_embeds=True because this is a guidance-distilled dev variant.
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
    """Available Flux1-Dev-DedistilledMixTuned-v4 model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """Flux1-Dev-DedistilledMixTuned-v4 model loader for text-to-image generation tasks."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name=_REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None
        self.guidance_scale = 3.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Flux1-Dev-DedistilledMixTuned-v4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _make_local_config_dir(self) -> str:
        """Create a temporary directory with the transformer config.json."""
        config_dir = tempfile.mkdtemp()
        transformer_dir = os.path.join(config_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)
        with open(os.path.join(transformer_dir, "config.json"), "w") as f:
            json.dump(_TRANSFORMER_CONFIG, f)
        return config_dir

    def _load_transformer(
        self, dtype: torch.dtype = torch.bfloat16
    ) -> FluxTransformer2DModel:
        """Load the transformer from the single-file fp8 safetensors checkpoint."""
        model_path = hf_hub_download(repo_id=_REPO_ID, filename=_FILENAME)
        config_dir = self._make_local_config_dir()

        self._transformer = FluxTransformer2DModel.from_single_file(
            model_path,
            config=config_dir,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
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
