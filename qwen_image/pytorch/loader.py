# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image component loader (QwenImagePipeline, 1328x1328 native resolution).

Components:
  - TextEncoder  -> Qwen2_5_VLForConditionalGeneration  (~8.3B, text path only)
  - Transformer  -> QwenImageTransformer2DModel         (~20B MMDiT denoiser)
  - Vae          -> AutoencoderKLQwenImage decoder       (~0.25B, 3D video VAE)
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
    IMG_SHAPE,
    MESH_NAMES,
    MESH_SHAPES,
    REPO_ID,
    QwenImageTransformerWrapper,
    QwenImageVAEDecoderWrapper,
    QwenTextEncoderWrapper,
    load_text_encoder,
    load_transformer,
    load_vae,
    make_packed_latents,
    make_prompt_embeds,
    make_vae_latents,
    shard_text_encoder_specs,
    shard_transformer_specs,
    tokenize_prompt,
)


class ModelVariant(StrEnum):
    """Loadable components of the Qwen-Image pipeline."""

    TEXT_ENCODER = "TextEncoder"
    TRANSFORMER = "Transformer"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Load individual Qwen-Image components without instantiating QwenImagePipeline."""

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
            model="Qwen-Image",
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
            return QwenTextEncoderWrapper(load_text_encoder(dtype)).eval()
        if self._variant == ModelVariant.TRANSFORMER:
            return QwenImageTransformerWrapper(
                load_transformer(dtype), img_shapes=[[IMG_SHAPE]]
            ).eval()
        if self._variant == ModelVariant.VAE:
            return QwenImageVAEDecoderWrapper(load_vae(dtype)).eval()

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        The VAE fits on a single chip, so any count maps to (1, 1).
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
        """Return tensor -> partition_spec dict for the active component."""
        if self._variant == ModelVariant.TEXT_ENCODER:
            return shard_text_encoder_specs(model.text_encoder)
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_transformer_specs(model.transformer)
        if self._variant == ModelVariant.VAE:
            return None
        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return synthetic inputs for the active component at 1328x1328."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            input_ids, attention_mask = tokenize_prompt()
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.TRANSFORMER:
            hidden_states = make_packed_latents(dtype)
            encoder_hidden_states, encoder_hidden_states_mask = make_prompt_embeds(dtype)
            # Pipeline passes scheduler timestep / 1000 (here a mid-noise step).
            timestep = torch.tensor([0.5], dtype=dtype)
            return [
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                timestep,
            ]

        if self._variant == ModelVariant.VAE:
            return [make_vae_latents(dtype)]

        raise ValueError(f"Unknown variant: {self._variant}")
