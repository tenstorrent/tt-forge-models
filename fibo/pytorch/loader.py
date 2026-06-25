# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FIBO (briaai/FIBO) component loader.

FIBO is BRIA AI's gated, 8B-parameter DiT flow-matching text-to-image model
(custom ``BriaFiboPipeline``; paper arXiv:2511.06876). It is a multi-component
pipeline, so — mirroring ``omnigen`` — each independently compilable component
is exposed as its own ``ModelVariant`` rather than one whole-pipeline graph:

  - TextEncoder → SmolLM3ForCausalLM (prompt → 4096-dim encoder hidden states)
  - Transformer → BriaFiboTransformer2DModel (Flux-style MMDiT denoiser — the
                  heavy per-step compute and the bringup's primary target)
  - VaeDecoder  → AutoencoderKLWan decoder (latent → image)

The scheduler / denoising loop / latent glue stay in host Python (the composite
generation step), matching the diffusion bringup pattern.

The HF repo is license-gated (``bria-fibo``); accept the license and
authenticate via ``HF_TOKEN`` before loading.

Reference: https://huggingface.co/briaai/FIBO
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
    TextEncoderWrapper,
    TransformerWrapper,
    VAEDecoderWrapper,
    load_text_encoder,
    load_text_encoder_inputs,
    load_transformer,
    load_transformer_inputs,
    load_vae,
    load_vae_decoder_inputs,
)


class ModelVariant(StrEnum):
    """Independently loadable components of the FIBO pipeline."""

    TEXT_ENCODER = "TextEncoder"
    TRANSFORMER = "Transformer"
    VAE_DECODER = "VaeDecoder"


class ModelLoader(ForgeModel):
    """Load individual FIBO components without driving the full pipeline."""

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.VAE_DECODER: ModelConfig(pretrained_model_name=REPO_ID),
    }

    # The denoiser is the heavy compute and the component that *must* run on
    # device, so it is the default variant.
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FIBO",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the wrapped component for this variant.

        Returns:
            TEXT_ENCODER → TextEncoderWrapper (SmolLM3, last-2 hidden states cat)
            TRANSFORMER  → TransformerWrapper (BriaFiboTransformer2DModel)
            VAE_DECODER  → VAEDecoderWrapper (AutoencoderKLWan decoder)
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return TextEncoderWrapper(load_text_encoder(dtype)).eval()
        if self._variant == ModelVariant.TRANSFORMER:
            return TransformerWrapper(load_transformer(dtype)).eval()
        if self._variant == ModelVariant.VAE_DECODER:
            return VAEDecoderWrapper(load_vae(dtype)).eval()

        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a list of synthetic native-resolution inputs for this variant.

        TEXT_ENCODER → [input_ids, attention_mask]
        TRANSFORMER  → [hidden_states, timestep, encoder_hidden_states,
                        layers_stacked, txt_ids, img_ids, attention_mask]
        VAE_DECODER  → [z (1, 48, 1, 64, 64)]
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return load_text_encoder_inputs(dtype)
        if self._variant == ModelVariant.TRANSFORMER:
            return load_transformer_inputs(dtype)
        if self._variant == ModelVariant.VAE_DECODER:
            return load_vae_decoder_inputs(dtype)

        raise ValueError(f"Unknown variant: {self._variant}")
