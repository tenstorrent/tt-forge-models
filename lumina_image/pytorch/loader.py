# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Lumina-Image-2.0 component loader.

Each variant corresponds to one independently loadable component:
  - TextEncoder  → Gemma2Model               (text_encoder subfolder)
  - Transformer  → Lumina2Transformer2DModel (transformer subfolder, wrapped for plain-tensor forward)
  - Vae          → VAEDecoderWrapper         (vae subfolder, decoder-only)
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
    LUMINA_REPO_ID,
    MESH_NAMES,
    MESH_SHAPES,
    Gemma2TextEncoderWrapper,
    Lumina2TransformerWrapper,
    VAEDecoderWrapper,
    load_text_encoder,
    load_transformer,
    load_vae,
    load_text_encoder_inputs,
    load_transformer_inputs,
    load_vae_inputs,
    shard_text_encoder_specs,
    shard_transformer_specs,
    shard_vae_specs,
)


class ModelVariant(StrEnum):
    """Loadable components of the Lumina-Image-2.0 pipeline."""

    TEXT_ENCODER = "TextEncoder"
    TRANSFORMER = "Transformer"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Load individual Lumina-Image-2.0 components without pulling the full pipeline.

    All three components share the same Alpha-VLLM/Lumina-Image-2.0 repo
    (text_encoder / transformer / vae subfolders).
    """

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=LUMINA_REPO_ID),
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=LUMINA_REPO_ID),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=LUMINA_REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant == ModelVariant.TEXT_ENCODER
            else ModelTask.MM_IMAGE_TTT
        )
        return ModelInfo(
            model="LuminaImage2",
            variant=variant,
            group=ModelGroup.RED,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module.

        Returns:
            TEXT_ENCODER → Gemma2TextEncoderWrapper (plain-tensor forward)
            TRANSFORMER  → Lumina2TransformerWrapper (plain-tensor forward)
            VAE          → VAEDecoderWrapper (decoder-only, returns plain tensor)
        """
        model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return Gemma2TextEncoderWrapper(load_text_encoder(model_name, dtype))
        if self._variant == ModelVariant.TRANSFORMER:
            return Lumina2TransformerWrapper(load_transformer(model_name, dtype))
        if self._variant == ModelVariant.VAE:
            return VAEDecoderWrapper(load_vae(model_name, dtype))

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        Supported device counts: 1, 2, 4, 8, 32.
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
          TEXT_ENCODER → Gemma2TextEncoderWrapper  (specs built from .encoder)
          TRANSFORMER  → Lumina2TransformerWrapper  (specs built from .transformer)
          VAE          → VAEDecoderWrapper          (specs built from .vae)
        """
        if self._variant == ModelVariant.TEXT_ENCODER:
            return shard_text_encoder_specs(model.encoder)
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_transformer_specs(model.transformer)
        if self._variant == ModelVariant.VAE:
            return shard_vae_specs(model.vae)
        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a list of synthetic input tensors for the active component.

        TEXT_ENCODER  → [input_ids (1,256) int64, attention_mask (1,256) int64]
        TRANSFORMER   → [hidden_states (1,16,128,128) bf16, timestep (1,) f32,
                         encoder_hidden_states (1,256,2304) bf16,
                         encoder_attention_mask (1,256) int64]
        VAE           → [z (1,16,128,128) bf16]
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return load_text_encoder_inputs(dtype)
        if self._variant == ModelVariant.TRANSFORMER:
            return load_transformer_inputs(dtype)
        if self._variant == ModelVariant.VAE:
            return load_vae_inputs(dtype)

        raise ValueError(f"Unknown variant: {self._variant}")
