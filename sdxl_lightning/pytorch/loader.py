# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SDXL-Lightning (SDXL-based, distilled) component loader.

SDXL-Lightning publishes only a distilled UNet checkpoint
(ByteDance/SDXL-Lightning). The remaining components are reused unchanged
from the base SDXL repo (stabilityai/stable-diffusion-xl-base-1.0).

Each variant corresponds to one independently loadable component:
  - TextEncoder   → CLIPTextModel                params=0.123B
  - TextEncoder2  → CLIPTextModelWithProjection  params=0.695B
  - Unet          → UNet2DConditionModel         params=2.567B   (Lightning weights)
  - Vae           → AutoencoderKL                params=0.084B
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
    CROSS_ATTN_DIM,
    CLIP_VOCAB_SIZE,
    DTYPE,
    LATENT_CHANNELS,
    LATENT_H,
    LATENT_W,
    POOLED_TEXT_EMBED_DIM,
    SDXL_BASE_REPO_ID,
    TEXT_SEQ_LEN,
    TIME_IDS_DIM,
    TextEncoder2Wrapper,
    TextEncoderWrapper,
    UNet2DConditionWrapper,
    VAEDecoderWrapper,
    load_text_encoder,
    load_text_encoder_2,
    load_unet,
    load_vae,
)


class ModelVariant(StrEnum):
    """Loadable components of the SDXL-Lightning pipeline."""

    TEXT_ENCODER = "TextEncoder"
    TEXT_ENCODER_2 = "TextEncoder2"
    UNET = "Unet"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Load individual SDXL-Lightning components without pulling the full pipeline.

    text_encoder / text_encoder_2 / vae weights come from
    stabilityai/stable-diffusion-xl-base-1.0; the unet uses the same config
    but the state_dict comes from ByteDance/SDXL-Lightning.
    """

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=SDXL_BASE_REPO_ID),
        ModelVariant.TEXT_ENCODER_2: ModelConfig(
            pretrained_model_name=SDXL_BASE_REPO_ID
        ),
        ModelVariant.UNET: ModelConfig(pretrained_model_name=SDXL_BASE_REPO_ID),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=SDXL_BASE_REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.UNET

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant in (ModelVariant.TEXT_ENCODER, ModelVariant.TEXT_ENCODER_2)
            else ModelTask.CONDITIONAL_GENERATION
        )
        return ModelInfo(
            model="SDXLLightning",
            variant=variant,
            group=ModelGroup.RED,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module.

        Returns:
            TEXT_ENCODER    → TextEncoderWrapper        (returns hidden_states[-2])
            TEXT_ENCODER_2  → TextEncoder2Wrapper       (returns (hidden_states[-2], text_embeds))
            UNET            → UNet2DConditionWrapper    (returns noise_pred, Lightning weights)
            VAE             → VAEDecoderWrapper         (returns decoded image)
        """
        model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return TextEncoderWrapper(load_text_encoder(model_name, dtype))
        if self._variant == ModelVariant.TEXT_ENCODER_2:
            return TextEncoder2Wrapper(load_text_encoder_2(model_name, dtype))
        if self._variant == ModelVariant.UNET:
            return UNet2DConditionWrapper(load_unet(model_name, dtype))
        if self._variant == ModelVariant.VAE:
            return VAEDecoderWrapper(load_vae(model_name, dtype))

        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a list of synthetic input tensors for the active component.

        TEXT_ENCODER    → [input_ids (1,77) int64]
        TEXT_ENCODER_2  → [input_ids (1,77) int64]
        UNET            → [sample (1,4,128,128), timestep scalar, encoder_hidden_states (1,77,2048),
                           text_embeds (1,1280), time_ids (1,6)]
        VAE             → [z (1,4,128,128)]
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant in (ModelVariant.TEXT_ENCODER, ModelVariant.TEXT_ENCODER_2):
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
            text_embeds = torch.randn(1, POOLED_TEXT_EMBED_DIM, dtype=dtype)
            time_ids = torch.randn(1, TIME_IDS_DIM, dtype=dtype)
            return [sample, timestep, encoder_hidden_states, text_embeds, time_ids]

        if self._variant == ModelVariant.VAE:
            z = torch.randn(1, LATENT_CHANNELS, LATENT_H, LATENT_W, dtype=dtype)
            return [z]

        raise ValueError(f"Unknown variant: {self._variant}")
