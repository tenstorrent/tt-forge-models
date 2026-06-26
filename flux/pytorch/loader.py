# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-dev component loader (1024x1024 native resolution).

Components:
  - ClipTextEncoder → CLIPTextModel          (pooled text embedding)
  - T5TextEncoder   → T5EncoderModel         (sequence text embedding)
  - Transformer     → FluxTransformer2DModel (single denoise step)
  - Vae             → AutoencoderKL decoder
"""

from typing import Optional

import torch

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
from .src.model_utils import (
    DTYPE,
    GUIDANCE_SCALE,
    LATENT_GRID_H,
    LATENT_GRID_W,
    MAX_SEQUENCE_LENGTH,
    REPO_ID,
    ClipTextEncoderWrapper,
    FluxTransformerWrapper,
    FluxVAEDecoderWrapper,
    T5TextEncoderWrapper,
    load_clip_text_encoder,
    load_t5_text_encoder,
    load_transformer,
    load_vae,
    make_packed_latents,
    make_pooled_embeds,
    make_prompt_embeds,
    make_vae_decoder_input,
    prepare_latent_image_ids,
    prepare_text_ids,
    tokenize_clip,
    tokenize_t5,
)


class ModelVariant(StrEnum):
    """Loadable components of the FLUX.1-dev pipeline."""

    CLIP_TEXT_ENCODER = "ClipTextEncoder"
    T5_TEXT_ENCODER = "T5TextEncoder"
    TRANSFORMER = "Transformer"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Load individual FLUX.1-dev components without instantiating FluxPipeline."""

    _VARIANTS = {
        ModelVariant.CLIP_TEXT_ENCODER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.T5_TEXT_ENCODER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    _TEXT_VARIANTS = (ModelVariant.CLIP_TEXT_ENCODER, ModelVariant.T5_TEXT_ENCODER)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant in cls._TEXT_VARIANTS
            else ModelTask.MM_IMAGE_TTT
        )
        return ModelInfo(
            model="FLUX",
            variant=variant,
            group=ModelGroup.RED,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.CLIP_TEXT_ENCODER:
            return ClipTextEncoderWrapper(load_clip_text_encoder(dtype)).eval()
        if self._variant == ModelVariant.T5_TEXT_ENCODER:
            return T5TextEncoderWrapper(load_t5_text_encoder(dtype)).eval()
        if self._variant == ModelVariant.TRANSFORMER:
            return FluxTransformerWrapper(load_transformer(dtype)).eval()
        if self._variant == ModelVariant.VAE:
            return FluxVAEDecoderWrapper(load_vae(dtype)).eval()

        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return synthetic inputs for the active component at 1024x1024."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.CLIP_TEXT_ENCODER:
            return [tokenize_clip()]

        if self._variant == ModelVariant.T5_TEXT_ENCODER:
            return [tokenize_t5()]

        if self._variant == ModelVariant.TRANSFORMER:
            hidden_states = make_packed_latents(dtype)
            pooled_projections = make_pooled_embeds(dtype)
            encoder_hidden_states = make_prompt_embeds(dtype)
            timestep = torch.tensor([1.0], dtype=dtype)
            guidance = torch.full([1], GUIDANCE_SCALE, dtype=dtype)
            txt_ids = prepare_text_ids(MAX_SEQUENCE_LENGTH, dtype)
            img_ids = prepare_latent_image_ids(LATENT_GRID_H, LATENT_GRID_W, dtype)
            return [
                hidden_states,
                timestep,
                guidance,
                pooled_projections,
                encoder_hidden_states,
                txt_ids,
                img_ids,
            ]

        if self._variant == ModelVariant.VAE:
            return [make_vae_decoder_input(dtype)]

        raise ValueError(f"Unknown variant: {self._variant}")
