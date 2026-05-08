# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HunyuanVideo 1.5 (480p t2v distilled) component loader.

Each variant corresponds to one independently loadable component:
  - TextEncoder   → Qwen2.5-VL encoder (text_encoder)       params=7.07B
  - TextEncoder2  → ByT5 encoder (text_encoder_2)            params=0.22B
  - Transformer   → HunyuanVideo15TransformerWrapper (DiT)   params=8.33B
  - Vae           → VAEDecoderWrapper (decoder-only)         params=1.26B
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
    BYT5_VOCAB_SIZE,
    DTYPE,
    IMAGE_EMBED_DIM,
    IMAGE_EMBED_SEQ,
    LATENT_H,
    LATENT_W,
    MESH_NAMES,
    MESH_SHAPES,
    NUM_CHANNELS_LATENTS,
    NUM_LATENT_FRAMES,
    QWEN_VOCAB_SIZE,
    TEXT_EMBED_2_DIM,
    TEXT_EMBED_DIM,
    TEXT_TOKEN_2_MAX_LEN,
    TEXT_TOKEN_MAX_LEN,
    TRANSFORMER_IN_CHANNELS,
    TRANSFORMER_TEXT_2_SEQ,
    TRANSFORMER_TEXT_SEQ,
    HunyuanVideo15TransformerWrapper,
    VAEDecoderWrapper,
    load_text_encoder,
    load_text_encoder_2,
    load_transformer,
    load_vae,
    shard_text_encoder_specs,
    shard_transformer_specs,
)

_REPO_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled"


class ModelVariant(StrEnum):
    """Loadable components of the HunyuanVideo 1.5 pipeline."""

    TEXT_ENCODER = "TextEncoder"
    TEXT_ENCODER_2 = "TextEncoder2"
    TRANSFORMER = "Transformer"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Load individual HunyuanVideo 1.5 components without pulling the full pipeline."""

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=_REPO_ID),
        ModelVariant.TEXT_ENCODER_2: ModelConfig(pretrained_model_name=_REPO_ID),
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=_REPO_ID),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=_REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant in (ModelVariant.TEXT_ENCODER, ModelVariant.TEXT_ENCODER_2)
            else ModelTask.MM_VIDEO_TTT
        )
        return ModelInfo(
            model="HunyuanVideo15",
            variant=variant,
            group=ModelGroup.RED,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module.

        Returns:
            TEXT_ENCODER   → Qwen2.5-VL encoder model
            TEXT_ENCODER_2 → ByT5 encoder model
            TRANSFORMER    → HunyuanVideo15TransformerWrapper
            VAE            → VAEDecoderWrapper (decoder-only, returns plain tensor)
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return load_text_encoder(dtype)
        if self._variant == ModelVariant.TEXT_ENCODER_2:
            return load_text_encoder_2(dtype)
        if self._variant == ModelVariant.TRANSFORMER:
            return HunyuanVideo15TransformerWrapper(load_transformer(dtype)).eval()
        if self._variant == ModelVariant.VAE:
            return VAEDecoderWrapper(load_vae(dtype)).eval()

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        Supported device counts: 1, 2, 4, 8, 32.
        TEXT_ENCODER_2 and VAE fit on a single chip so any count maps to (1, 1).
        """
        if self._variant in (ModelVariant.TEXT_ENCODER_2, ModelVariant.VAE):
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
          TEXT_ENCODER   → Qwen2.5-VL encoder
          TEXT_ENCODER_2 → None (single-chip, no sharding)
          TRANSFORMER    → HunyuanVideo15TransformerWrapper (specs from .transformer)
          VAE            → None (single-chip, no sharding)
        """
        if self._variant == ModelVariant.TEXT_ENCODER:
            return shard_text_encoder_specs(model)
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_transformer_specs(model.transformer)
        return None

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a list of synthetic input tensors for the active component.

        TEXT_ENCODER   → [input_ids (1,1108) int64, attention_mask (1,1108) int64]
        TEXT_ENCODER_2 → [input_ids (1,256) int64, attention_mask (1,256) float32]
        TRANSFORMER    → [hidden_states, timestep, encoder_hidden_states,
                          encoder_attention_mask, encoder_hidden_states_2,
                          encoder_attention_mask_2, image_embeds]
        VAE            → [z (1,32,5,30,53) bfloat16]
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            input_ids = torch.randint(
                0, QWEN_VOCAB_SIZE, (1, TEXT_TOKEN_MAX_LEN), dtype=torch.long
            )
            attention_mask = torch.ones(1, TEXT_TOKEN_MAX_LEN, dtype=torch.long)
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.TEXT_ENCODER_2:
            input_ids = torch.randint(
                0, BYT5_VOCAB_SIZE, (1, TEXT_TOKEN_2_MAX_LEN), dtype=torch.long
            )
            # Pipeline calls text_encoder_2 with attention_mask.float()
            attention_mask = torch.ones(1, TEXT_TOKEN_2_MAX_LEN, dtype=torch.float32)
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.TRANSFORMER:
            hidden_states = torch.randn(
                1,
                TRANSFORMER_IN_CHANNELS,
                NUM_LATENT_FRAMES,
                LATENT_H,
                LATENT_W,
                dtype=dtype,
            )
            timestep = torch.tensor([1000.0], dtype=dtype)
            encoder_hidden_states = torch.randn(
                1, TRANSFORMER_TEXT_SEQ, TEXT_EMBED_DIM, dtype=dtype
            )
            encoder_attention_mask = torch.ones(1, TRANSFORMER_TEXT_SEQ, dtype=dtype)
            encoder_hidden_states_2 = torch.randn(
                1, TRANSFORMER_TEXT_2_SEQ, TEXT_EMBED_2_DIM, dtype=dtype
            )
            encoder_attention_mask_2 = torch.ones(
                1, TRANSFORMER_TEXT_2_SEQ, dtype=dtype
            )
            image_embeds = torch.zeros(1, IMAGE_EMBED_SEQ, IMAGE_EMBED_DIM, dtype=dtype)
            return [
                hidden_states,
                timestep,
                encoder_hidden_states,
                encoder_attention_mask,
                encoder_hidden_states_2,
                encoder_attention_mask_2,
                image_embeds,
            ]

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
