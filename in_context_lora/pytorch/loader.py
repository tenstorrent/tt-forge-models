# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
In-Context LoRA for FLUX.1-dev model loader implementation.

Loads the black-forest-labs/FLUX.1-dev base pipeline and applies task-specific
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

import os
from typing import Any, Optional

import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

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

BASE_MODEL = "camenduru/FLUX.1-dev-diffusers"
LORA_REPO = "ali-vilab/In-Context-LoRA"

# In-Context LoRA uses paired image generation at double width
_HEIGHT = 128
_WIDTH = 256  # double width for side-by-side paired images
_GUIDANCE_SCALE = 3.5


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
        self.pipeline: Optional[FluxPipeline] = None

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

    def _load_pipeline_random_weights(self, dtype: torch.dtype) -> FluxPipeline:
        """Construct FluxPipeline from configs with random weights (no large downloads)."""
        model_name = self._variant_config.pretrained_model_name

        transformer_config = FluxTransformer2DModel.load_config(
            model_name, subfolder="transformer"
        )
        transformer = FluxTransformer2DModel.from_config(transformer_config).to(dtype)

        from transformers import T5Config

        t5_config = T5Config.from_pretrained(model_name, subfolder="text_encoder_2")
        text_encoder_2 = T5EncoderModel(t5_config).to(dtype)

        text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=dtype
        )
        vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae", torch_dtype=dtype
        )
        tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            model_name, subfolder="tokenizer_2"
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )

        return FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=transformer,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the FLUX transformer with In-Context LoRA weights applied.

        Returns:
            FluxTransformer2DModel (with LoRA merged when not in random-weights mode).
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        # Use random weights when explicitly requested or in compile-only mode
        # (compile-only validates model architecture, not weight values)
        use_random = bool(os.environ.get("TT_RANDOM_WEIGHTS")) or bool(
            os.environ.get("TT_COMPILE_ONLY_SYSTEM_DESC")
        )

        if use_random:
            self.pipeline = self._load_pipeline_random_weights(dtype)
        else:
            # hf-xet segfaults in this environment; disable it for downloads
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            import huggingface_hub.constants as hf_constants

            hf_constants.HF_HUB_DISABLE_XET = True

            self.pipeline = FluxPipeline.from_pretrained(
                self._variant_config.pretrained_model_name,
                torch_dtype=dtype,
                **kwargs,
            )

            lora_file = _LORA_FILES[self._variant]
            self.pipeline.load_lora_weights(
                LORA_REPO,
                weight_name=lora_file,
            )

        return self.pipeline.transformer

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> Any:
        """Prepare direct transformer inputs for In-Context LoRA generation.

        In-Context LoRA uses paired (side-by-side) image generation, so the
        canvas width is doubled relative to a standard single-image generation.

        Returns:
            dict of tensors for FluxTransformer2DModel.forward().
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        prompt = (
            "This two-part image portrays a couple of cartoon cats in detective "
            "attire; [LEFT] a black cat in a trench coat and fedora holds a "
            "magnifying glass and peers to the right, while [RIGHT] a white cat "
            "with a bow tie and matching hat raises an eyebrow in curiosity."
        )
        max_sequence_length = 256
        do_classifier_free_guidance = _GUIDANCE_SCALE > 1.0
        num_images_per_prompt = 1
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4

        # CLIP text encoding
        text_inputs_clip = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        pooled_prompt_embeds = self.pipeline.text_encoder(
            text_inputs_clip.input_ids, output_hidden_states=False
        ).pooler_output.to(dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            batch_size, num_images_per_prompt
        ).view(batch_size * num_images_per_prompt, -1)

        # T5 text encoding
        text_inputs_t5 = self.pipeline.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        prompt_embeds = self.pipeline.text_encoder_2(
            text_inputs_t5.input_ids, output_hidden_states=False
        )[0].to(dtype=dtype)
        _, seq_len_t5, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(batch_size, num_images_per_prompt, 1).view(
            batch_size * num_images_per_prompt, seq_len_t5, -1
        )

        text_ids = torch.zeros(prompt_embeds.shape[1], 3, dtype=dtype)

        height_latent = 2 * (int(_HEIGHT) // (self.pipeline.vae_scale_factor * 2))
        width_latent = 2 * (int(_WIDTH) // (self.pipeline.vae_scale_factor * 2))
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
        latents = latents.permute(0, 2, 4, 1, 3, 5).reshape(
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

        guidance = (
            torch.full([batch_size], _GUIDANCE_SCALE, dtype=dtype)
            if do_classifier_free_guidance
            else None
        )

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
