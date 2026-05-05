# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HunyuanVideo component loader.

Each variant corresponds to one independently loadable component:
  - TextEncoder   → LLaMA encoder (text_encoder)
  - TextEncoder2  → CLIP encoder (text_encoder_2)
  - Transformer   → HunyuanVideoTransformerWrapper (DiT)
  - Vae           → VAEDecoderWrapper (decoder-only)
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
    MESH_NAMES,
    MESH_SHAPES,
    HunyuanVideoTransformerWrapper,
    VAEDecoderWrapper,
    load_text_encoder,
    load_text_encoder_2,
    load_text_encoder_2_inputs,
    load_text_encoder_inputs,
    load_transformer,
    load_transformer_inputs,
    load_vae,
    load_vae_inputs,
    shard_text_encoder_2_specs,
    shard_text_encoder_specs,
    shard_transformer_specs,
    shard_vae_specs,
)


class ModelVariant(StrEnum):
    """Loadable components of the HunyuanVideo pipeline."""

    TEXT_ENCODER = "TextEncoder"
    TEXT_ENCODER_2 = "TextEncoder2"
    TRANSFORMER = "Transformer"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Load individual HunyuanVideo components without pulling the full pipeline."""

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.TEXT_ENCODER_2: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=REPO_ID),
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
            model="HunyuanVideo",
            variant=variant,
            group=ModelGroup.RED,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module.

        Returns:
            TEXT_ENCODER   → LLaMA encoder model
            TEXT_ENCODER_2 → CLIP text encoder model
            TRANSFORMER    → HunyuanVideoTransformerWrapper
            VAE            → VAEDecoderWrapper (decoder-only, returns plain tensor)
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return load_text_encoder(dtype)
        if self._variant == ModelVariant.TEXT_ENCODER_2:
            return load_text_encoder_2(dtype)
        if self._variant == ModelVariant.TRANSFORMER:
            return HunyuanVideoTransformerWrapper(load_transformer(dtype)).eval()
        if self._variant == ModelVariant.VAE:
            return VAEDecoderWrapper(load_vae(dtype)).eval()

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        Supported device counts: 1, 2, 4, 8, 32. All four components are
        sharded so that the full pipeline can run on a single mesh.
        """
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return tensor → partition_spec dict for the active component.

        Expects the same model object returned by load_model():
          TEXT_ENCODER   → LLaMA encoder
          TEXT_ENCODER_2 → CLIP text encoder
          TRANSFORMER    → HunyuanVideoTransformerWrapper (specs from .transformer)
          VAE            → VAEDecoderWrapper (specs from .vae)
        """
        if self._variant == ModelVariant.TEXT_ENCODER:
            return shard_text_encoder_specs(model)
        if self._variant == ModelVariant.TEXT_ENCODER_2:
            return shard_text_encoder_2_specs(model)
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_transformer_specs(model.transformer)
        if self._variant == ModelVariant.VAE:
            return shard_vae_specs(model.vae)
        return None

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a list of synthetic input tensors for the active component.

        TEXT_ENCODER   → [input_ids (1,256) int64, attention_mask (1,256) int64]
        TEXT_ENCODER_2 → [input_ids (1,77) int64, attention_mask (1,77) int64]
        TRANSFORMER    → [hidden_states, timestep, encoder_hidden_states,
                          encoder_attention_mask, pooled_projections]
        VAE            → [z (1,16,2,16,16) bfloat16]
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return load_text_encoder_inputs(dtype)
        if self._variant == ModelVariant.TEXT_ENCODER_2:
            return load_text_encoder_2_inputs(dtype)
        if self._variant == ModelVariant.TRANSFORMER:
            return load_transformer_inputs(dtype)
        if self._variant == ModelVariant.VAE:
            return load_vae_inputs(dtype)

        raise ValueError(f"Unknown variant: {self._variant}")
