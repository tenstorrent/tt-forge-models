# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nunchaku FLUX.1-Kontext-dev quantized model loader implementation.

Loads a 4-bit SVDQuant quantized FLUX.1-Kontext-dev transformer via nunchaku
and injects it into the standard diffusers FluxKontextPipeline for
image-to-image editing.

Available variants:
- INT4: SVD-quantized INT4 weights (pre-Blackwell GPUs)
"""

from typing import Optional

import torch
from diffusers import AutoencoderTiny, FluxKontextPipeline  # type: ignore[import]
from nunchaku import NunchakuFluxTransformer2dModel  # type: ignore[import]

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

BASE_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
NUNCHAKU_REPO = "nunchaku-ai/nunchaku-flux.1-kontext-dev"


class ModelVariant(StrEnum):
    """Available Nunchaku FLUX.1-Kontext-dev variants."""

    INT4 = "INT4"


class ModelLoader(ForgeModel):
    """Nunchaku FLUX.1-Kontext-dev quantized model loader."""

    _VARIANTS = {
        ModelVariant.INT4: ModelConfig(
            pretrained_model_name=NUNCHAKU_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.INT4

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None
        self.guidance_scale = 2.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NUNCHAKU_FLUX_KONTEXT_DEV",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load FluxKontextPipeline with nunchaku quantized transformer."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            NUNCHAKU_REPO,
            filename="svdq-int4_r32-flux.1-kontext-dev.safetensors",
        )

        self.pipe = FluxKontextPipeline.from_pretrained(
            BASE_MODEL,
            transformer=transformer,
            torch_dtype=dtype,
        )

        self.pipe.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taef1",
            torch_dtype=dtype,
        )

        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_tiling()

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the nunchaku quantized FLUX.1-Kontext-dev transformer.

        Returns:
            torch.nn.Module: The quantized FLUX Kontext transformer model instance.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Load and return sample inputs for the nunchaku FLUX.1-Kontext-dev model.

        Returns:
            dict: Input tensors for the transformer model.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        max_sequence_length = 256
        prompt = "Make the background a sunset over mountains"
        num_images_per_prompt = 1
        height = 128
        width = 128
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        num_channels_latents = self.pipe.transformer.config.in_channels // 4

        # Text encoding for CLIP
        text_inputs_clip = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_clip = text_inputs_clip.input_ids
        pooled_prompt_embeds = self.pipe.text_encoder(
            text_input_ids_clip, output_hidden_states=False
        ).pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            batch_size, num_images_per_prompt
        )
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            batch_size * num_images_per_prompt, -1
        )

        # Text encoding for T5
        text_inputs_t5 = self.pipe.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_t5 = text_inputs_t5.input_ids
        prompt_embeds = self.pipe.text_encoder_2(
            text_input_ids_t5, output_hidden_states=False
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        _, seq_len_t5, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(batch_size, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len_t5, -1
        )

        # Create text IDs
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=dtype)

        # Create latents
        height_latent = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))

        shape = (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent,
            width_latent,
        )

        # FLUX Kontext concatenates reference image latents with noise latents
        noise_latents = torch.randn(shape, dtype=dtype)
        image_latents = torch.randn(shape, dtype=dtype)
        combined_latents = torch.cat([noise_latents, image_latents], dim=2)

        combined_height = combined_latents.shape[2]
        combined_latents = combined_latents.view(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            combined_height // 2,
            2,
            width_latent // 2,
            2,
        )
        combined_latents = combined_latents.permute(0, 2, 4, 1, 3, 5)
        combined_latents = combined_latents.reshape(
            batch_size * num_images_per_prompt,
            (combined_height // 2) * (width_latent // 2),
            num_channels_latents * 4,
        )

        # Prepare latent image IDs
        latent_image_ids = torch.zeros(combined_height // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(combined_height // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        # Prepare guidance
        guidance = torch.full([batch_size], self.guidance_scale, dtype=dtype)

        return {
            "hidden_states": combined_latents,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }
