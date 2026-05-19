# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Krea Realtime Video 14B component loader.

Each variant corresponds to one independently loadable component:
  - TextEncoder  → UMT5EncoderModel    (text_encoder subfolder)  params=6.73B
  - Transformer  → CausalWanWrapper    (transformer subfolder, wrapped for simple forward)  params=14.29B
  - Vae          → VAEDecoderWrapper   (vae subfolder, decoder-only)  params=0.13B
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
    LATENT_H,
    LATENT_W,
    MAX_SEQ_LEN,
    MESH_NAMES,
    MESH_SHAPES,
    NUM_CHANNELS_LATENTS,
    NUM_FRAMES_PER_BLOCK,
    NUM_LATENT_FRAMES,
    TEXT_EMBED_DIM,
    UMT5_VOCAB_SIZE,
    CausalWanWrapper,
    VAEDecoderWrapper,
    load_text_encoder,
    load_transformer,
    load_vae,
    shard_text_encoder_specs,
    shard_transformer_specs,
)

_KREA_REPO = "krea/krea-realtime-video"
_WAN_REPO = "Wan-AI/Wan2.1-T2V-14B-Diffusers"


class ModelVariant(StrEnum):
    """Loadable components of the Krea Realtime Video 14B pipeline."""

    TEXT_ENCODER = "TextEncoder"
    TRANSFORMER = "Transformer"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Load individual Krea Realtime Video components without pulling the full pipeline.

    The krea model reuses text_encoder and vae from Wan-AI/Wan2.1-T2V-14B-Diffusers;
    only the transformer weights come from krea/krea-realtime-video.
    See: krea/krea-realtime-video/modular_model_index.json
    """

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=_WAN_REPO),
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=_KREA_REPO),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=_WAN_REPO),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant == ModelVariant.TEXT_ENCODER
            else ModelTask.MM_VIDEO_TTT
        )
        return ModelInfo(
            model="KreaRealtimeVideo",
            variant=variant,
            group=ModelGroup.RED,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module.

        Returns:
            TEXT_ENCODER → UMT5EncoderModel
            TRANSFORMER  → CausalWanWrapper (raw CausalWanModel with simplified forward)
            VAE          → VAEDecoderWrapper (decoder-only, returns plain tensor)
        """
        model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return load_text_encoder(model_name, dtype)
        if self._variant == ModelVariant.TRANSFORMER:
            return CausalWanWrapper(load_transformer(model_name, dtype))
        if self._variant == ModelVariant.VAE:
            return VAEDecoderWrapper(load_vae(model_name, dtype))

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        Supported device counts: 1, 2, 4, 8, 32.
        VAE fits on a single chip so any count maps to (1, 1).
        """
        if self._variant == ModelVariant.VAE:
            return (1, 1), MESH_NAMES

        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return tensor → partition_spec dict for the active component.

        Expects the same model object returned by load_model():
          TEXT_ENCODER → UMT5EncoderModel
          TRANSFORMER  → CausalWanWrapper  (specs built from .transformer)
          VAE          → None (single-chip, no sharding)
        """
        if self._variant == ModelVariant.TEXT_ENCODER:
            return shard_text_encoder_specs(model)
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_transformer_specs(model.transformer)
        if self._variant == ModelVariant.VAE:
            return None
        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a list of synthetic input tensors for the active component.

        TEXT_ENCODER  → [input_ids (1,512) int64, attention_mask (1,512) int64]
        TRANSFORMER   → [x (1,16,3,60,104) bf16, t (1,3) f32, context (1,512,4096) bf16]
        VAE           → [z (1,16,3,60,104) bf16]
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            input_ids = torch.randint(
                0, UMT5_VOCAB_SIZE, (1, MAX_SEQ_LEN), dtype=torch.long
            )
            attention_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.TRANSFORMER:
            x = torch.randn(
                1,
                NUM_CHANNELS_LATENTS,
                NUM_LATENT_FRAMES,
                LATENT_H,
                LATENT_W,
                dtype=dtype,
            )
            t = torch.full((1, NUM_FRAMES_PER_BLOCK), 1000.0, dtype=torch.float32)
            context = torch.randn(1, MAX_SEQ_LEN, TEXT_EMBED_DIM, dtype=dtype)
            return [x, t, context]

        if self._variant == ModelVariant.VAE:
            z = torch.randn(
                1,
                NUM_CHANNELS_LATENTS,
                NUM_LATENT_FRAMES,
                LATENT_H,
                LATENT_W,
                dtype=dtype,
            )
            return [z]

        raise ValueError(f"Unknown variant: {self._variant}")
