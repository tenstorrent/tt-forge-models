# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-Image component loader.

Each variant corresponds to one independently loadable component:
  - TextEncoder            → T5 encoder (text_encoder)
  - VisionLanguageEncoder  → GlmImageForConditionalGeneration (vision_language_encoder)
  - Transformer            → GlmImageTransformerWrapper (DiT)
  - Vae                    → VAEDecoderWrapper (decoder-only)
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
    GlmImageTransformerWrapper,
    GlmImageVisionLanguageWrapper,
    VAEDecoderWrapper,
    load_text_encoder,
    load_text_encoder_inputs,
    load_transformer,
    load_transformer_inputs,
    load_vae,
    load_vae_inputs,
    load_vision_language_encoder,
    load_vision_language_encoder_inputs,
    shard_text_encoder_specs,
    shard_transformer_specs,
    shard_vae_specs,
    shard_vision_language_encoder_specs,
)


class ModelVariant(StrEnum):
    """Loadable components of the GLM-Image pipeline."""

    TEXT_ENCODER = "TextEncoder"
    VISION_LANGUAGE_ENCODER = "VisionLanguageEncoder"
    TRANSFORMER = "Transformer"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Load individual GLM-Image components without pulling the full pipeline."""

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.VISION_LANGUAGE_ENCODER: ModelConfig(
            pretrained_model_name=REPO_ID
        ),
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        if variant == ModelVariant.TEXT_ENCODER:
            task = ModelTask.NLP_EMBED_GEN
        elif variant == ModelVariant.VISION_LANGUAGE_ENCODER:
            task = ModelTask.MM_CONDITIONAL_GENERATION
        else:
            task = ModelTask.MM_IMAGE_TTT
        return ModelInfo(
            model="GLMImage",
            variant=variant,
            group=ModelGroup.RED,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module.

        Returns:
            TEXT_ENCODER            → T5 encoder model
            VISION_LANGUAGE_ENCODER → GlmImageVisionLanguageWrapper
            TRANSFORMER             → GlmImageTransformerWrapper
            VAE                     → VAEDecoderWrapper (decoder-only, returns plain tensor)
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return load_text_encoder(dtype)
        if self._variant == ModelVariant.VISION_LANGUAGE_ENCODER:
            return GlmImageVisionLanguageWrapper(
                load_vision_language_encoder(dtype)
            ).eval()
        if self._variant == ModelVariant.TRANSFORMER:
            return GlmImageTransformerWrapper(load_transformer(dtype)).eval()
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
          TEXT_ENCODER            → T5 encoder
          VISION_LANGUAGE_ENCODER → GlmImageVisionLanguageWrapper (specs from .vlm)
          TRANSFORMER             → GlmImageTransformerWrapper (specs from .transformer)
          VAE                     → VAEDecoderWrapper (specs from .vae)
        """
        if self._variant == ModelVariant.TEXT_ENCODER:
            return shard_text_encoder_specs(model)
        if self._variant == ModelVariant.VISION_LANGUAGE_ENCODER:
            return shard_vision_language_encoder_specs(model.vlm)
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_transformer_specs(model.transformer)
        if self._variant == ModelVariant.VAE:
            return shard_vae_specs(model.vae)
        return None

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a list of synthetic input tensors for the active component.

        TEXT_ENCODER            → [input_ids (16,38) int64, attention_mask (16,38) int64]
        VISION_LANGUAGE_ENCODER → [input_ids (1,394) int64, attention_mask (1,394) int64,
                                   pixel_values (1392,768) dtype, image_grid_thw (2,3) int64,
                                   images_per_sample (1,) int64, cache_position (394,) int64,
                                   logits_to_keep scalar int64]
        TRANSFORMER             → [hidden_states (1,16,128,144), encoder_hidden_states (1,376,1472),
                                   prior_token_id (1,4608) int64, prior_token_drop (1,4608) bool,
                                   timestep ([999.]), target_size ([[1024.,1152.]] bf16),
                                   crop_coords ([[0.,0.]] bf16)]
        VAE                     → [z (1,16,128,144) dtype]
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return load_text_encoder_inputs(dtype)
        if self._variant == ModelVariant.VISION_LANGUAGE_ENCODER:
            return load_vision_language_encoder_inputs(dtype)
        if self._variant == ModelVariant.TRANSFORMER:
            return load_transformer_inputs(dtype)
        if self._variant == ModelVariant.VAE:
            return load_vae_inputs(dtype)

        raise ValueError(f"Unknown variant: {self._variant}")
