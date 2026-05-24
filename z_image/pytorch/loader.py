# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image component loader (Tongyi-MAI/Z-Image).

Variants: VAE decoder, Qwen3 text encoder, ZImageTransformer2DModel (DiT).
All components run on a single chip.
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
    PROMPT,
    Qwen3TextEncoderWrapper,
    REPO_ID,
    VAEDecoderWrapper,
    ZImageTransformerWrapper,
    encode_prompt_hidden_states,
    load_text_encoder,
    load_transformer,
    load_vae,
    make_latent_inputs,
    tokenize_prompt,
)


class ModelVariant(StrEnum):
    """Loadable components of the Z-Image pipeline."""

    TEXT_ENCODER = "TextEncoder"
    TRANSFORMER = "Transformer"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Load individual Z-Image components without instantiating ZImagePipeline."""

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    _TEXT_VARIANTS = (ModelVariant.TEXT_ENCODER,)

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
            model="ZImage",
            variant=variant,
            group=ModelGroup.RED,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        dtype = dtype_override if dtype_override is not None else DTYPE
        if self._variant == ModelVariant.TEXT_ENCODER:
            return Qwen3TextEncoderWrapper(load_text_encoder(dtype)).eval()
        if self._variant == ModelVariant.TRANSFORMER:
            return ZImageTransformerWrapper(load_transformer(dtype)).eval()
        if self._variant == ModelVariant.VAE:
            return VAEDecoderWrapper(load_vae(dtype)).eval()
        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            input_ids, attention_mask = tokenize_prompt(PROMPT)
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.TRANSFORMER:
            latents = make_latent_inputs(dtype)
            timestep = torch.tensor([0.5], dtype=dtype)
            encoder = load_text_encoder(dtype)
            cap_feats = encode_prompt_hidden_states(encoder, PROMPT, dtype=dtype)
            return [latents, timestep, cap_feats]

        if self._variant == ModelVariant.VAE:
            return [make_latent_inputs(dtype)]

        raise ValueError(f"Unknown variant: {self._variant}")
