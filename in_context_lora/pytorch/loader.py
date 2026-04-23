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
        """Load the FLUX.1-dev pipeline with In-Context LoRA weights applied.

        Returns:
            FluxPipeline with LoRA weights merged.
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

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for text-to-image generation.

        Returns:
            dict with prompt key.
        """
        if prompt is None:
            prompt = (
                "This two-part image portrays a couple of cartoon cats in detective "
                "attire; [LEFT] a black cat in a trench coat and fedora holds a "
                "magnifying glass and peers to the right, while [RIGHT] a white cat "
                "with a bow tie and matching hat raises an eyebrow in curiosity."
            )

        return {
            "prompt": prompt,
        }
