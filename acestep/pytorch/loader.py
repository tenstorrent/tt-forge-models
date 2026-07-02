# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ACE-Step 1.5 component loader (text-to-music, 30 s native clip).

ACE-Step 1.5 (https://huggingface.co/ACE-Step/Ace-Step1.5) is a multi-stage
generative music model. Like a diffusion image pipeline it is not a single
forward pass, so each independently-compilable component is exposed as its own
variant:

  - TextEncoder -> Qwen3-Embedding-0.6B   (prompt conditioning encoder)
  - Lm          -> acestep-5Hz-lm-1.7B    (Qwen3 semantic-token LM)
  - Denoiser    -> AceStepDiTModel        (per-step flow-matching DiT; key component)
  - VaeDecoder  -> AutoencoderOobleck      (48 kHz stereo audio decoder)

Requires `vector_quantize_pytorch` and `einops` (imported by the vendored
turbo modeling file). See requirements.txt next to this loader.
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
    REPO_ID,
    load_denoiser,
    load_lm,
    load_text_encoder,
    load_vae_decoder,
    make_denoiser_inputs,
    make_lm_inputs,
    make_text_encoder_inputs,
    make_vae_inputs,
)


class ModelVariant(StrEnum):
    """Loadable components of the ACE-Step 1.5 pipeline."""

    TEXT_ENCODER = "TextEncoder"
    LM = "Lm"
    DENOISER = "Denoiser"
    VAE_DECODER = "VaeDecoder"


class ModelLoader(ForgeModel):
    """Load individual ACE-Step 1.5 components as standalone torch.nn.Modules."""

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.LM: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.DENOISER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.VAE_DECODER: ModelConfig(pretrained_model_name=REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.DENOISER

    _TEXT_VARIANTS = (ModelVariant.TEXT_ENCODER, ModelVariant.LM)

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant in cls._TEXT_VARIANTS
            else ModelTask.MM_AUDIO_TTT
        )
        return ModelInfo(
            model="ACE-Step-1.5",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return load_text_encoder(dtype)
        if self._variant == ModelVariant.LM:
            return load_lm(dtype)
        if self._variant == ModelVariant.DENOISER:
            return load_denoiser(dtype)
        if self._variant == ModelVariant.VAE_DECODER:
            return load_vae_decoder(dtype)

        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return synthetic inputs for the active component at the native clip."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return make_text_encoder_inputs()
        if self._variant == ModelVariant.LM:
            return make_lm_inputs()
        if self._variant == ModelVariant.DENOISER:
            return make_denoiser_inputs(dtype)
        if self._variant == ModelVariant.VAE_DECODER:
            return make_vae_inputs(dtype)

        raise ValueError(f"Unknown variant: {self._variant}")
