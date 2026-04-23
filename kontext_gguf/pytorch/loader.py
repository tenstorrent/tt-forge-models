# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
calcuis/kontext-gguf model loader implementation for image-to-image generation.

Loads the calcuis/kontext-gguf GGUF-quantized FLUX.1-Kontext-dev transformer using
diffusers' GGUF quantization support.

Repository:
- https://huggingface.co/calcuis/kontext-gguf
"""
import torch
from diffusers import GGUFQuantizationConfig
from diffusers.models import FluxTransformer2DModel
from huggingface_hub import hf_hub_download
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


class ModelVariant(StrEnum):
    """Available calcuis/kontext-gguf model variants."""

    Q4_0 = "Q4_0"
    Q8_0 = "Q8_0"


class ModelLoader(ForgeModel):
    """calcuis/kontext-gguf model loader implementation for image-to-image tasks."""

    _VARIANTS = {
        ModelVariant.Q4_0: ModelConfig(
            pretrained_model_name="calcuis/kontext-gguf",
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name="calcuis/kontext-gguf",
        ),
    }

    _GGUF_FILES = {
        ModelVariant.Q4_0: "flux1-kontext-dev-f32-q4_0.gguf",
        ModelVariant.Q8_0: "flux1-kontext-dev-f32-q8_0.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.Q4_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None
        self.guidance_scale = 2.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Kontext-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        repo_id = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        gguf_path = hf_hub_download(repo_id=repo_id, filename=gguf_file)

        self.transformer = FluxTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
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

        # Create latent tensor and pack to (B, H*W, C)
        latents = torch.randn(
            batch_size, num_channels_latents * 4, h_packed, w_packed, dtype=dtype
        )
        latents = latents.reshape(batch_size, num_channels_latents * 4, -1).permute(
            0, 2, 1
        )

        # Prepare latent image IDs
        latent_image_ids = torch.zeros(h_packed, w_packed, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(h_packed)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(w_packed)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        # Prompt embeddings: use random tensors matching joint_attention_dim
        max_sequence_length = 256
        joint_attention_dim = config.joint_attention_dim
        prompt_embeds = torch.randn(
            batch_size, max_sequence_length, joint_attention_dim, dtype=dtype
        )

        # Pooled projections
        pooled_prompt_embeds = torch.randn(
            batch_size, config.pooled_projection_dim, dtype=dtype
        )

        # Text IDs
        text_ids = torch.zeros(max_sequence_length, 3).to(dtype=dtype)

        # Guidance
        guidance = torch.full([batch_size], self.guidance_scale, dtype=dtype)

        # Timestep
        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

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
