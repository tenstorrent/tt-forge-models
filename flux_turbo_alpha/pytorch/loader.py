# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-Turbo-Alpha model loader implementation for text-to-image generation.

Loads the FLUX.1-dev base pipeline and applies the alimama-creative/FLUX.1-Turbo-Alpha
distilled LoRA weights for fast 8-step text-to-image generation.
"""
import os

import torch
from diffusers import FluxPipeline, AutoencoderTiny
from diffusers.models import FluxTransformer2DModel
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

BASE_MODEL = "black-forest-labs/FLUX.1-dev"
LORA_REPO = "alimama-creative/FLUX.1-Turbo-Alpha"

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
    """Available FLUX.1-Turbo-Alpha model variants."""

    TURBO_ALPHA = "TurboAlpha"


class ModelLoader(ForgeModel):
    """FLUX.1-Turbo-Alpha model loader for text-to-image generation tasks."""

    _VARIANTS = {
        ModelVariant.TURBO_ALPHA: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TURBO_ALPHA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None
        self._transformer = None
        self.guidance_scale = 3.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="FLUX.1-Turbo-Alpha",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _is_random_weights():
        return os.environ.get("TT_RANDOM_WEIGHTS", "") == "1"

    def _load_transformer_from_config(self, dtype_override=None):
        """Construct the FLUX transformer from config with random weights."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self._transformer = FluxTransformer2DModel(**_TRANSFORMER_CONFIG).to(dtype)
        return self._transformer

    def _load_pipeline(self, dtype_override=None):
        """Load the FLUX.1-dev pipeline and apply Turbo-Alpha LoRA weights."""
        token = os.environ.get("HF_TOKEN")
        pipe_kwargs = {
            "use_safetensors": True,
        }
        if token:
            pipe_kwargs["token"] = token
        if dtype_override is not None:
            pipe_kwargs["torch_dtype"] = dtype_override

        self.pipe = FluxPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, **pipe_kwargs
        )

        self.pipe.load_lora_weights(LORA_REPO)
        self.pipe.fuse_lora()

        vae_kwargs = {}
        if dtype_override is not None:
            vae_kwargs["torch_dtype"] = dtype_override

        self.pipe.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taef1", **vae_kwargs
        )

        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_tiling()

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FLUX transformer model with Turbo-Alpha LoRA applied.

        Returns:
            torch.nn.Module: The FLUX transformer model instance.
        """
        if self._is_random_weights():
            if self._transformer is None:
                self._load_transformer_from_config(dtype_override=dtype_override)
            if dtype_override is not None:
                self._transformer = self._transformer.to(dtype_override)
            return self._transformer

        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the FLUX model.

        Returns:
            dict: Input tensors that can be fed to the transformer model.
        """
        max_sequence_length = 256
        do_classifier_free_guidance = self.guidance_scale > 1.0
        height = 128
        width = 128
        num_images_per_prompt = 1
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        vae_scale_factor = 8

        if self._is_random_weights():
            if self._transformer is None:
                self._load_transformer_from_config(dtype_override=dtype_override)
            num_channels_latents = self._transformer.config.in_channels // 4
            pooled_projection_dim = self._transformer.config.pooled_projection_dim
            joint_attention_dim = self._transformer.config.joint_attention_dim

            pooled_prompt_embeds = torch.randn(
                batch_size * num_images_per_prompt,
                pooled_projection_dim,
                dtype=dtype,
            )
            prompt_embeds = torch.randn(
                batch_size * num_images_per_prompt,
                max_sequence_length,
                joint_attention_dim,
                dtype=dtype,
            )
            text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)
        else:
            if self.pipe is None:
                self._load_pipeline(dtype_override=dtype_override)
            num_channels_latents = self.pipe.transformer.config.in_channels // 4
            vae_scale_factor = self.pipe.vae_scale_factor

            prompt = "A DSLR photo of a shiny VW van parked in a sunny meadow"

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

            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=dtype)

        height_latent = 2 * (int(height) // (vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (vae_scale_factor * 2))

        shape = (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent,
            width_latent,
        )

        latents = torch.randn(shape, dtype=dtype)
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

        if do_classifier_free_guidance:
            guidance = torch.full([batch_size], self.guidance_scale, dtype=dtype)
        else:
            guidance = None

        inputs = {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }

        return inputs
