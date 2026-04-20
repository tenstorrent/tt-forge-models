# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tiny Random Flex.2-preview model loader implementation for text-to-image generation.
"""
import torch
from diffusers import FluxPipeline
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
    """Available Tiny Random Flex.2-preview model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """Tiny Random Flex.2-preview model loader for text-to-image generation tasks."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="katuni4ka/tiny-random-flex.2-preview",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Tiny Random Flex.2-preview",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        pipe_kwargs = {}
        if dtype_override is not None:
            pipe_kwargs["torch_dtype"] = dtype_override

        self.pipe = FluxPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, **pipe_kwargs
        )
        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        # Configuration
        max_sequence_length = 256
        prompt = "a cat sitting on a couch"
        height = 32
        width = 32
        num_images_per_prompt = 1
        dtype = dtype_override if dtype_override is not None else torch.float32

        # Flex.2 transformer expects packed channels composed of:
        #   noise latents + inpaint latents + mask + control latents.
        # in_channels is the packed channel count (total * 4 after 2x2 patching);
        # the control stack contributes a fixed 33 pre-pack channels
        # (16 inpaint + 1 mask + 16 control).
        total_pre_pack_channels = self.pipe.transformer.config.in_channels // 4
        control_pre_pack_channels = 33
        num_channels_latents = total_pre_pack_channels - control_pre_pack_channels

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

        # Compute latent dimensions
        vae_scale_factor = self.pipe.vae_scale_factor
        height_latent = 2 * (int(height) // (vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (vae_scale_factor * 2))

        def _pack(tensor):
            batch, channels, h, w = tensor.shape
            packed = tensor.view(batch, channels, h // 2, 2, w // 2, 2)
            packed = packed.permute(0, 2, 4, 1, 3, 5)
            return packed.reshape(batch, (h // 2) * (w // 2), channels * 4)

        # Noise latents (the portion being denoised)
        noise_latents = torch.randn(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height_latent,
            width_latent,
            dtype=dtype,
        )
        noise_latents = _pack(noise_latents)

        # Inpaint reference latents (16 channels)
        inpaint_latents = torch.randn(
            batch_size * num_images_per_prompt,
            16,
            height_latent,
            width_latent,
            dtype=dtype,
        )
        inpaint_latents = _pack(inpaint_latents)

        # Mask latents (single channel)
        mask_latents = torch.ones(
            batch_size * num_images_per_prompt,
            1,
            height_latent,
            width_latent,
            dtype=dtype,
        )
        mask_latents = _pack(mask_latents)

        # Control image latents (16 channels)
        control_latents = torch.randn(
            batch_size * num_images_per_prompt,
            16,
            height_latent,
            width_latent,
            dtype=dtype,
        )
        control_latents = _pack(control_latents)

        hidden_states = torch.cat(
            [noise_latents, inpaint_latents, mask_latents, control_latents], dim=-1
        )

        # Prepare latent image IDs
        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        # Flex.2-preview transformer has guidance_embeds=True
        guidance = torch.full([batch_size * num_images_per_prompt], 3.5, dtype=dtype)

        inputs = {
            "hidden_states": hidden_states,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }

        return inputs
