# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Arturmel/perfect1 model loader implementation.

Arturmel/perfect1 is a LoRA fine-tune of FLUX.1-dev that applies a
"perfection style" aesthetic to text-to-image generation.

Available variants:
- DEFAULT: Arturmel/perfect1 LoRA applied on top of FLUX.1-dev
"""

from typing import Optional

import torch
from diffusers import FluxPipeline, AutoencoderTiny

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

# FLUX.1-dev spatial downsampling factor and packing config
_VAE_SCALE_FACTOR = 8
_TOKENIZER_MAX_LENGTH = 77
_T5_MAX_SEQ_LEN = 256
_CLIP_POOLED_DIM = 768
_T5_HIDDEN_DIM = 4096


class ModelVariant(StrEnum):
    """Available Arturmel/perfect1 model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """Arturmel/perfect1 LoRA model loader implementation."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="Arturmel/perfect1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None
        self._transformer = None
        self._use_random_weights = False
        self.guidance_scale = 3.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Arturmel/perfect1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        from huggingface_hub.errors import GatedRepoError

        pipe_kwargs = {
            "use_safetensors": True,
        }
        if dtype_override is not None:
            pipe_kwargs["torch_dtype"] = dtype_override

        try:
            self.pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", **pipe_kwargs
            )

            self.pipe.load_lora_weights(
                self._variant_config.pretrained_model_name,
                weight_name="perfection style v1.safetensors",
            )

            vae_kwargs = {}
            if dtype_override is not None:
                vae_kwargs["torch_dtype"] = dtype_override

            self.pipe.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taef1", **vae_kwargs
            )

            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_tiling()
        except GatedRepoError:
            self._load_transformer_random(dtype_override)

        return self.pipe

    def _load_transformer_random(self, dtype_override=None):
        """Create FluxTransformer2DModel with FLUX.1-dev config and random weights."""
        from diffusers import FluxTransformer2DModel

        self._transformer = FluxTransformer2DModel(
            patch_size=2,
            guidance_embeds=True,
        )
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype_override)
        self._use_random_weights = True

    def load_model(self, *, dtype_override=None, **kwargs):
        if self._transformer is None and self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if self._use_random_weights:
            if dtype_override is not None:
                self._transformer = self._transformer.to(dtype_override)
            return self._transformer

        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self._transformer is None and self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if self._use_random_weights:
            return self._make_random_inputs(dtype_override, batch_size)

        return self._make_pipeline_inputs(dtype_override, batch_size)

    def _make_random_inputs(self, dtype_override=None, batch_size=1):
        """Create inputs directly with known FLUX.1-dev shapes (no pipeline needed)."""
        height = 128
        width = 128
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        num_channels_latents = self._transformer.config.in_channels // 4

        height_latent = 2 * (int(height) // (_VAE_SCALE_FACTOR * 2))
        width_latent = 2 * (int(width) // (_VAE_SCALE_FACTOR * 2))

        seq_len = (height_latent // 2) * (width_latent // 2)
        packed_channels = num_channels_latents * 4

        latents = torch.randn(batch_size, seq_len, packed_channels, dtype=dtype)

        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        do_classifier_free_guidance = self.guidance_scale > 1.0
        if do_classifier_free_guidance:
            guidance = torch.full([batch_size], self.guidance_scale, dtype=dtype)
        else:
            guidance = None

        return {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": guidance,
            "pooled_projections": torch.randn(
                batch_size, _CLIP_POOLED_DIM, dtype=dtype
            ),
            "encoder_hidden_states": torch.randn(
                batch_size, _T5_MAX_SEQ_LEN, _T5_HIDDEN_DIM, dtype=dtype
            ),
            "txt_ids": torch.zeros(_T5_MAX_SEQ_LEN, 3, dtype=dtype),
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }

    def _make_pipeline_inputs(self, dtype_override=None, batch_size=1):
        """Create inputs using the loaded pipeline components."""
        max_sequence_length = 256
        prompt = "A beautiful portrait in perfection style"
        do_classifier_free_guidance = self.guidance_scale > 1.0
        height = 128
        width = 128
        num_images_per_prompt = 1
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        num_channels_latents = self.pipe.transformer.config.in_channels // 4

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

        if do_classifier_free_guidance:
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
