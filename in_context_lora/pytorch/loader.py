# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
In-Context LoRA for FLUX.1-dev model loader implementation.

Loads the black-forest-labs/FLUX.1-dev base pipeline and applies task-specific
In-Context LoRA weights from ali-vilab/In-Context-LoRA for multi-task
text-to-image generation with customizable intrinsic relationships.

When FLUX.1-dev is inaccessible (gated repo), falls back to a
FluxTransformer2DModel with random weights and the correct FLUX.1-dev
architecture, with In-Context LoRA weights applied directly.

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

from typing import Any, Optional

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers.loaders import FluxLoraLoaderMixin
from huggingface_hub.errors import GatedRepoError

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
LORA_REPO = "ali-vilab/In-Context-LoRA"

# FLUX.1-dev transformer architecture parameters (used for random-weights fallback)
_FLUX_DEV_TRANSFORMER_CONFIG = dict(
    patch_size=1,
    in_channels=64,
    out_channels=64,
    num_layers=19,
    num_single_layers=38,
    attention_head_dim=128,
    num_attention_heads=24,
    joint_attention_dim=4096,
    pooled_projection_dim=768,
    guidance_embeds=True,
    axes_dims_rope=[16, 56, 56],
)


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
        self.transformer: Optional[FluxTransformer2DModel] = None
        # Set when GatedRepoError prevents loading the full pipeline
        self._transformer_only: bool = False

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

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the FLUX.1-dev pipeline with In-Context LoRA weights applied.

        Falls back to a FluxTransformer2DModel with random weights if the
        gated FLUX.1-dev repo is inaccessible.

        Returns:
            FluxPipeline with LoRA weights merged, or FluxTransformer2DModel
            with LoRA weights applied when the base model is inaccessible.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        lora_file = _LORA_FILES[self._variant]

        try:
            self.pipeline = FluxPipeline.from_pretrained(
                self._variant_config.pretrained_model_name,
                torch_dtype=dtype,
                **kwargs,
            )
            self.pipeline.load_lora_weights(LORA_REPO, weight_name=lora_file)
            self._transformer_only = False
            return self.pipeline

        except GatedRepoError:
            # Base model is gated — build transformer from config with random
            # weights and apply the publicly accessible LoRA weights directly.
            self._transformer_only = True

            transformer = FluxTransformer2DModel(**_FLUX_DEV_TRANSFORMER_CONFIG).to(
                dtype=dtype
            )
            transformer.eval()
            for param in transformer.parameters():
                param.requires_grad = False

            state_dict, network_alphas = FluxLoraLoaderMixin.lora_state_dict(
                LORA_REPO,
                weight_name=lora_file,
                return_alphas=True,
            )
            FluxLoraLoaderMixin.load_lora_into_transformer(
                state_dict,
                network_alphas=network_alphas,
                transformer=transformer,
                _pipeline=None,
            )

            self.transformer = transformer
            return transformer

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for the model.

        Returns a text prompt dict for the full pipeline, or synthetic tensor
        inputs matching FluxTransformer2DModel.forward when in fallback mode.
        """
        if not self._transformer_only:
            if prompt is None:
                prompt = (
                    "This two-part image portrays a couple of cartoon cats in detective "
                    "attire; [LEFT] a black cat in a trench coat and fedora holds a "
                    "magnifying glass and peers to the right, while [RIGHT] a white cat "
                    "with a bow tie and matching hat raises an eyebrow in curiosity."
                )
            return {"prompt": prompt}

        return self._make_transformer_inputs(dtype=torch.bfloat16)

    def _make_transformer_inputs(
        self,
        batch_size: int = 1,
        height: int = 128,
        width: int = 128,
        max_seq_len: int = 77,
        dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """Generate synthetic tensor inputs for FluxTransformer2DModel.forward."""
        vae_scale_factor = 8
        latent_h = height // vae_scale_factor
        latent_w = width // vae_scale_factor
        # 2x2 spatial packing applied before transformer
        h_packed = latent_h // 2
        w_packed = latent_w // 2
        num_img_tokens = h_packed * w_packed
        in_channels = _FLUX_DEV_TRANSFORMER_CONFIG["in_channels"]
        joint_attention_dim = _FLUX_DEV_TRANSFORMER_CONFIG["joint_attention_dim"]
        pooled_projection_dim = _FLUX_DEV_TRANSFORMER_CONFIG["pooled_projection_dim"]
        guidance_embeds = _FLUX_DEV_TRANSFORMER_CONFIG["guidance_embeds"]

        hidden_states = torch.randn(
            batch_size, num_img_tokens, in_channels, dtype=dtype
        )
        encoder_hidden_states = torch.randn(
            batch_size, max_seq_len, joint_attention_dim, dtype=dtype
        )
        pooled_projections = torch.randn(batch_size, pooled_projection_dim, dtype=dtype)

        img_ids = torch.zeros(num_img_tokens, 3, dtype=dtype)
        img_ids[:, 1] = torch.arange(num_img_tokens, dtype=dtype) // w_packed
        img_ids[:, 2] = torch.arange(num_img_tokens, dtype=dtype) % w_packed

        txt_ids = torch.zeros(max_seq_len, 3, dtype=dtype)
        timestep = torch.tensor([1.0] * batch_size, dtype=dtype)
        guidance = (
            torch.full([batch_size], 3.5, dtype=dtype) if guidance_embeds else None
        )

        inputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
        }
        if guidance is not None:
            inputs["guidance"] = guidance
        return inputs
