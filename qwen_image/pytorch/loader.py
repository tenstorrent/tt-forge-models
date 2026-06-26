# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen/Qwen-Image component loader (128x128 bringup resolution).

Components (QwenImagePipeline):
  - TextEncoder  -> Qwen2_5_VLForConditionalGeneration  (~8B)
  - Transformer  -> QwenImageTransformer2DModel          (~20B)
  - Vae          -> AutoencoderKLQwenImage decoder       (~0.13B)
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
    IMG_SHAPES,
    MESH_NAMES,
    MESH_SHAPES,
    PROMPT,
    QwenImageTextEncoderWrapper,
    QwenImageTransformerWrapper,
    QwenImageVAEDecoderWrapper,
    REPO_ID,
    load_text_encoder,
    load_transformer,
    load_vae,
    make_packed_latents,
    make_prompt_embeds,
    make_prompt_embeds_mask,
    make_vae_decoder_input,
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
            model="QwenImage",
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
            return QwenImageTextEncoderWrapper(load_text_encoder(dtype)).eval()
        if self._variant == ModelVariant.TRANSFORMER:
            return QwenImageTransformerWrapper(load_transformer(dtype), IMG_SHAPES).eval()
        if self._variant == ModelVariant.VAE:
            return QwenImageVAEDecoderWrapper(load_vae(dtype)).eval()

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        The VAE decoder fits on a single chip, so any device count maps to (1, 1).
        Text encoder and transformer shard weights across all chips on "model".
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
        """Return synthetic inputs for the active component at the bringup resolution."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            input_ids, attention_mask = tokenize_prompt(PROMPT)
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.TRANSFORMER:
            hidden_states = make_packed_latents(dtype)
            encoder_hidden_states = make_prompt_embeds(dtype)
            encoder_hidden_states_mask = make_prompt_embeds_mask()
            # Pipeline passes scheduler t / 1000 as the timestep.
            timestep = torch.tensor([500.0 / 1000.0], dtype=dtype)
            return [
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                timestep,
            ]

        if self._variant == ModelVariant.VAE:
            return [make_vae_decoder_input(dtype)]

        raise ValueError(f"Unknown variant: {self._variant}")
