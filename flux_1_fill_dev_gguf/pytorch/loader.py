# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-Fill-dev GGUF model loader implementation for image inpainting.

This loader uses GGUF-quantized variants of the FLUX.1-Fill-dev model from
YarvixPA/FLUX.1-Fill-dev-GGUF. The GGUF transformer is loaded via diffusers'
FluxTransformer2DModel.from_single_file and plugged into a FluxFillPipeline
built from the original black-forest-labs/FLUX.1-Fill-dev repository.
"""

import os
from typing import Optional

import torch
from diffusers import FluxFillPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
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

GGUF_REPO = "YarvixPA/FLUX.1-Fill-dev-GGUF"
BASE_REPO = "black-forest-labs/FLUX.1-Fill-dev"

# FLUX.1-Fill-dev transformer architecture parameters.
# in_channels=384: noisy latents (64) + masked image latents (64) + packed mask (256).
_TRANSFORMER_CONFIG = {
    "patch_size": 1,
    "in_channels": 384,
    "num_layers": 19,
    "num_single_layers": 38,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "pooled_projection_dim": 768,
    "axes_dims_rope": [16, 56, 56],
    "guidance_embeds": True,
}

# Known FLUX VAE scale factor and text encoder output dimensions.
_VAE_SCALE_FACTOR = 8
_CLIP_POOLED_DIM = 768
_T5_HIDDEN_DIM = 4096
_MAX_SEQUENCE_LENGTH = 256


class ModelVariant(StrEnum):
    """Available FLUX.1-Fill-dev GGUF quantization variants."""

    Q3_K_S = "Q3_K_S"
    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q4_K_S = "Q4_K_S"
    Q5_0 = "Q5_0"
    Q5_1 = "Q5_1"
    Q5_K_S = "Q5_K_S"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.Q3_K_S: "flux1-fill-dev-Q3_K_S.gguf",
    ModelVariant.Q4_0: "flux1-fill-dev-Q4_0.gguf",
    ModelVariant.Q4_1: "flux1-fill-dev-Q4_1.gguf",
    ModelVariant.Q4_K_S: "flux1-fill-dev-Q4_K_S.gguf",
    ModelVariant.Q5_0: "flux1-fill-dev-Q5_0.gguf",
    ModelVariant.Q5_1: "flux1-fill-dev-Q5_1.gguf",
    ModelVariant.Q5_K_S: "flux1-fill-dev-Q5_K_S.gguf",
    ModelVariant.Q6_K: "flux1-fill-dev-Q6_K.gguf",
    ModelVariant.Q8_0: "flux1-fill-dev-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """FLUX.1-Fill-dev GGUF model loader for image inpainting."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=GGUF_REPO) for variant in _GGUF_FILES
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K_S

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX.1-Fill-dev GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16):
        """Load the FluxFillPipeline with a GGUF-quantized transformer."""
        if os.environ.get("TT_RANDOM_WEIGHTS") or os.environ.get(
            "TT_COMPILE_ONLY_SYSTEM_DESC"
        ):
            self._transformer = FluxTransformer2DModel(**_TRANSFORMER_CONFIG).to(dtype)
            return

        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=gguf_file)

        transformer = FluxTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )

        self.pipe = FluxFillPipeline.from_pretrained(
            BASE_REPO,
            transformer=transformer,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        self._transformer = self.pipe.transformer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FLUX Fill transformer.

        Returns:
            torch.nn.Module: The FLUX Fill transformer model instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            self._load_pipeline(dtype)
        elif dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample inputs for the FLUX Fill transformer.

        The FLUX Fill transformer expects hidden_states that include the noisy latents
        concatenated with masked image latents and a packed mask along dim=2,
        resulting in in_channels of 384 (64 + 64 + 256).

        Returns:
            dict: Input tensors for the transformer model.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        height = 128
        width = 128
        num_images_per_prompt = 1

        if os.environ.get("TT_RANDOM_WEIGHTS") or os.environ.get(
            "TT_COMPILE_ONLY_SYSTEM_DESC"
        ):
            if self._transformer is None:
                self._load_pipeline(dtype)

            height_latent = 2 * (height // (_VAE_SCALE_FACTOR * 2))
            width_latent = 2 * (width // (_VAE_SCALE_FACTOR * 2))
            seq_len = (height_latent // 2) * (width_latent // 2)

            noise_channels = 64
            masked_image_channels = 64
            mask_channels = 256
            latents = torch.randn(
                batch_size * num_images_per_prompt, seq_len, noise_channels, dtype=dtype
            )
            masked_image_latents = torch.randn(
                batch_size * num_images_per_prompt,
                seq_len,
                masked_image_channels + mask_channels,
                dtype=dtype,
            )
            hidden_states = torch.cat((latents, masked_image_latents), dim=2)

            pooled_projections = torch.randn(
                batch_size * num_images_per_prompt, _CLIP_POOLED_DIM, dtype=dtype
            )
            encoder_hidden_states = torch.randn(
                batch_size * num_images_per_prompt,
                _MAX_SEQUENCE_LENGTH,
                _T5_HIDDEN_DIM,
                dtype=dtype,
            )
            text_ids = torch.zeros(_MAX_SEQUENCE_LENGTH, 3, dtype=dtype)

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
                "guidance": torch.tensor([3.5], dtype=dtype),
                "pooled_projections": pooled_projections,
                "encoder_hidden_states": encoder_hidden_states,
                "txt_ids": text_ids,
                "img_ids": latent_image_ids,
                "joint_attention_kwargs": {},
            }

        if self.pipe is None:
            self._load_pipeline(dtype)

        max_sequence_length = _MAX_SEQUENCE_LENGTH
        prompt = "A cat sitting on a windowsill"

        # CLIP text encoding
        text_inputs_clip = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        pooled_prompt_embeds = self.pipe.text_encoder(
            text_inputs_clip.input_ids, output_hidden_states=False
        ).pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            batch_size, num_images_per_prompt
        )
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            batch_size * num_images_per_prompt, -1
        )

        # T5 text encoding
        text_inputs_t5 = self.pipe.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        prompt_embeds = self.pipe.text_encoder_2(
            text_inputs_t5.input_ids, output_hidden_states=False
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        _, seq_len_t5, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(batch_size, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len_t5, -1
        )

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=dtype)

        # Latent dimensions
        height_latent = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))
        seq_len = (height_latent // 2) * (width_latent // 2)

        # Noisy latents (packed format)
        noise_channels = 64
        latents = torch.randn(
            batch_size * num_images_per_prompt, seq_len, noise_channels, dtype=dtype
        )

        # Masked image latents (VAE-encoded masked image + packed mask)
        masked_image_channels = 64
        mask_channels = 256
        masked_image_latents = torch.randn(
            batch_size * num_images_per_prompt,
            seq_len,
            masked_image_channels + mask_channels,
            dtype=dtype,
        )

        # Concatenate noisy latents with masked image latents along channel dim
        hidden_states = torch.cat((latents, masked_image_latents), dim=2)

        # Latent image IDs
        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        # FLUX.1-Fill-dev uses classifier-free guidance
        guidance = torch.tensor([3.5], dtype=dtype)

        return {
            "hidden_states": hidden_states,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
