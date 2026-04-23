# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
In-Context LoRA for FLUX.1-dev model loader implementation.

Loads the camenduru/FLUX.1-dev-ungated base pipeline and applies task-specific
In-Context LoRA weights from ali-vilab/In-Context-LoRA for multi-task
text-to-image generation with customizable intrinsic relationships.

Available variants:
- COUPLE_PROFILE: Couple profile design LoRA
- FILM_STORYBOARD: Film storyboard LoRA
- FONT_DESIGN: Font design LoRA
- HOME_DECORATION: Home decoration LoRA
- PORTRAIT_ILLUSTRATION: Portrait illustration LoRA
- PORTRAIT_PHOTOGRAPHY: Portrait photography LoRA
- PPT_TEMPLATES: PPT template LoRA
- SANDSTORM_VISUAL_EFFECT: Sandstorm visual effect LoRA
- SPARKLERS_VISUAL_EFFECT: Sparklers visual effect LoRA
- VISUAL_IDENTITY_DESIGN: Visual identity design LoRA
"""

from typing import Optional

import torch
from diffusers import AutoencoderTiny, FluxPipeline

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

BASE_MODEL = "camenduru/FLUX.1-dev-ungated"
LORA_REPO = "ali-vilab/In-Context-LoRA"


class ModelVariant(StrEnum):
    """Available In-Context LoRA task variants."""

    COUPLE_PROFILE = "couple-profile"
    FILM_STORYBOARD = "film-storyboard"
    FONT_DESIGN = "font-design"
    HOME_DECORATION = "home-decoration"
    PORTRAIT_ILLUSTRATION = "portrait-illustration"
    PORTRAIT_PHOTOGRAPHY = "portrait-photography"
    PPT_TEMPLATES = "ppt-templates"
    SANDSTORM_VISUAL_EFFECT = "sandstorm-visual-effect"
    SPARKLERS_VISUAL_EFFECT = "sparklers-visual-effect"
    VISUAL_IDENTITY_DESIGN = "visual-identity-design"


_LORA_FILES = {
    ModelVariant.COUPLE_PROFILE: "couple-profile.safetensors",
    ModelVariant.FILM_STORYBOARD: "film-storyboard.safetensors",
    ModelVariant.FONT_DESIGN: "font-design.safetensors",
    ModelVariant.HOME_DECORATION: "home-decoration.safetensors",
    ModelVariant.PORTRAIT_ILLUSTRATION: "portrait-illustration.safetensors",
    ModelVariant.PORTRAIT_PHOTOGRAPHY: "portrait-photography.safetensors",
    ModelVariant.PPT_TEMPLATES: "ppt-templates.safetensors",
    ModelVariant.SANDSTORM_VISUAL_EFFECT: "sandstorm-visual-effect.safetensors",
    ModelVariant.SPARKLERS_VISUAL_EFFECT: "sparklers-visual-effect.safetensors",
    ModelVariant.VISUAL_IDENTITY_DESIGN: "visual-identity-design.safetensors",
}


class ModelLoader(ForgeModel):
    """In-Context LoRA for FLUX.1-dev model loader."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=BASE_MODEL)
        for variant in ModelVariant
    }
    DEFAULT_VARIANT = ModelVariant.COUPLE_PROFILE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None
        self.guidance_scale = 3.5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="IN_CONTEXT_LORA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load FLUX pipeline and apply In-Context LoRA weights."""
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

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the FLUX transformer with In-Context LoRA weights applied.

        Returns:
            torch.nn.Module: The FLUX transformer model instance.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the FLUX transformer.

        Returns:
            dict: Input tensors for the transformer model.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        max_sequence_length = 256
        prompt = (
            "This two-part image portrays a couple of cartoon cats in detective "
            "attire; [LEFT] a black cat in a trench coat and fedora holds a "
            "magnifying glass and peers to the right, while [RIGHT] a white cat "
            "with a bow tie and matching hat raises an eyebrow in curiosity."
        )
        do_classifier_free_guidance = self.guidance_scale > 1.0
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

        # Prepare latent image IDs
        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        # Prepare guidance
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
