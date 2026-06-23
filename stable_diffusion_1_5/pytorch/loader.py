# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 1.5 component loader.

Stable Diffusion 1.5 (``stable-diffusion-v1-5/stable-diffusion-v1-5``) is a
latent-diffusion text-to-image pipeline. It is not a single forward pass — it
decomposes into independently compilable neural components plus host-Python
glue (PNDM scheduler + denoising loop). Each variant here corresponds to one
independently loadable component:

  - TextEncoder → CLIPTextModel            params=0.123B
  - Unet        → UNet2DConditionModel     params=0.860B   (denoiser)
  - Vae         → AutoencoderKL (decoder)  params=0.084B

``load_model`` returns the selected component as an ``nn.Module`` (the format
the tt-xla model tester / composite pipeline expects). The composite pipeline
test wires the components together with the scheduler in host Python.
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
    CLIP_VOCAB_SIZE,
    CROSS_ATTN_DIM,
    DTYPE,
    LATENT_CHANNELS,
    LATENT_H,
    LATENT_W,
    SD15_REPO_ID,
    TEXT_SEQ_LEN,
    TextEncoderWrapper,
    UNet2DConditionWrapper,
    VAEDecoderWrapper,
    load_text_encoder,
    load_unet,
    load_vae,
)


class ModelVariant(StrEnum):
    """Loadable components of the Stable Diffusion 1.5 pipeline."""

    TEXT_ENCODER = "TextEncoder"
    UNET = "Unet"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Load individual Stable Diffusion 1.5 components without pulling the full pipeline.

    All component weights come from
    ``stable-diffusion-v1-5/stable-diffusion-v1-5`` (text_encoder / unet / vae
    subfolders).
    """

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=SD15_REPO_ID),
        ModelVariant.UNET: ModelConfig(pretrained_model_name=SD15_REPO_ID),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=SD15_REPO_ID),
    }

    DEFAULT_VARIANT = ModelVariant.UNET

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with the requested variant.

        Args:
            variant: Optional ``ModelVariant``; falls back to ``DEFAULT_VARIANT``.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant == ModelVariant.TEXT_ENCODER
            else ModelTask.CONDITIONAL_GENERATION
        )
        return ModelInfo(
            model="Stable Diffusion 1.5",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a ``torch.nn.Module``.

        Returns:
            TEXT_ENCODER → TextEncoderWrapper       (returns last_hidden_state)
            UNET         → UNet2DConditionWrapper   (returns noise_pred)
            VAE          → VAEDecoderWrapper         (returns decoded image)
        """
        model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return TextEncoderWrapper(load_text_encoder(model_name, dtype))
        if self._variant == ModelVariant.UNET:
            return UNet2DConditionWrapper(load_unet(model_name, dtype))
        if self._variant == ModelVariant.VAE:
            return VAEDecoderWrapper(load_vae(model_name, dtype))

        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a list of synthetic input tensors for the active component.

        TEXT_ENCODER → [input_ids (1,77) int64]
        UNET         → [sample (1,4,64,64), timestep scalar, encoder_hidden_states (1,77,768)]
        VAE          → [z (1,4,64,64)]
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            input_ids = torch.randint(
                0, CLIP_VOCAB_SIZE, (1, TEXT_SEQ_LEN), dtype=torch.long
            )
            return [input_ids]

        if self._variant == ModelVariant.UNET:
            sample = torch.randn(1, LATENT_CHANNELS, LATENT_H, LATENT_W, dtype=dtype)
            timestep = torch.tensor(1.0, dtype=dtype)
            encoder_hidden_states = torch.randn(
                1, TEXT_SEQ_LEN, CROSS_ATTN_DIM, dtype=dtype
            )
            return [sample, timestep, encoder_hidden_states]

        if self._variant == ModelVariant.VAE:
            z = torch.randn(1, LATENT_CHANNELS, LATENT_H, LATENT_W, dtype=dtype)
            return [z]

        raise ValueError(f"Unknown variant: {self._variant}")
