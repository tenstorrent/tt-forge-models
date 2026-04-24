# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
strangerzonehf/Flux-Ultimate-LoRA-Collection model loader implementation.

Loads the FLUX.1-dev base pipeline and applies a style LoRA from
strangerzonehf/Flux-Ultimate-LoRA-Collection for stylized text-to-image
generation.

Repository: https://huggingface.co/strangerzonehf/Flux-Ultimate-LoRA-Collection
"""

from typing import Optional

import torch
from diffusers import AutoencoderTiny, FluxPipeline

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

BASE_MODEL = "camenduru/FLUX.1-dev-diffusers"
LORA_REPO = "strangerzonehf/Flux-Ultimate-LoRA-Collection"

LORA_ANIMEO = "Animeo.safetensors"
LORA_3D_REALISM = "3D-Realism.safetensors"
LORA_CASTOR_3D_PORTRAIT = "Castor-3D-Portrait-Flux-LoRA.safetensors"


class ModelVariant(StrEnum):
    """Available Flux-Ultimate-LoRA-Collection style variants."""

    ANIMEO = "Animeo"
    REALISM_3D = "3D-Realism"
    CASTOR_3D_PORTRAIT = "Castor-3D-Portrait"


_LORA_FILES = {
    ModelVariant.ANIMEO: LORA_ANIMEO,
    ModelVariant.REALISM_3D: LORA_3D_REALISM,
    ModelVariant.CASTOR_3D_PORTRAIT: LORA_CASTOR_3D_PORTRAIT,
}


class ModelLoader(ForgeModel):
    """strangerzonehf/Flux-Ultimate-LoRA-Collection model loader."""

    _VARIANTS = {
        ModelVariant.ANIMEO: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.REALISM_3D: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.CASTOR_3D_PORTRAIT: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.ANIMEO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe: Optional[FluxPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Flux-Ultimate-LoRA-Collection",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load FLUX.1-dev pipeline and apply the selected LoRA weights."""
        pipe_kwargs = {"use_safetensors": True}
        if dtype_override is not None:
            pipe_kwargs["torch_dtype"] = dtype_override

        self.pipe = FluxPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, **pipe_kwargs
        )

        lora_file = _LORA_FILES[self._variant]
        self.pipe.load_lora_weights(LORA_REPO, weight_name=lora_file)

        vae_kwargs = {}
        if dtype_override is not None:
            vae_kwargs["torch_dtype"] = dtype_override

        self.pipe.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taef1", **vae_kwargs
        )
        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_tiling()

        return self.pipe

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the FLUX.1-dev pipeline with the selected LoRA weights applied.

        Returns:
            torch.nn.Module: The FLUX transformer model instance.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare inputs for text-to-image generation.

        Returns:
            dict: Input tensors for the transformer model.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        max_sequence_length = 256
        prompt = "An astronaut riding a green horse"
        guidance_scale = 3.5
        height = 128
        width = 128
        num_images_per_prompt = 1
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

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=dtype)

        height_latent = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))

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

        guidance = torch.full([batch_size], guidance_scale, dtype=dtype)

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
