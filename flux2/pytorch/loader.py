# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2-dev component loader (1024x1024 output resolution).

Components:
  - TextEncoder  → Mistral3ForConditionalGeneration  (~24B)
  - Transformer  → Flux2Transformer2DModel           (~32B)
  - Vae          → AutoencoderKLFlux2 decoder        (~84M)
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
    GUIDANCE_SCALE,
    HEIGHT,
    LATENT_GRID_H,
    LATENT_GRID_W,
    MAX_SEQUENCE_LENGTH,
    MESH_NAMES,
    MESH_SHAPES,
    PROMPT,
    REPO_ID,
    WIDTH,
    Flux2TransformerWrapper,
    Flux2VAEDecoderWrapper,
    Mistral3TextEncoderWrapper,
    load_text_encoder,
    load_transformer,
    load_vae,
    make_packed_latents,
    make_synthetic_prompt_embeds,
    make_vae_decoder_input,
    prepare_latent_image_ids,
    prepare_text_ids,
    shard_text_encoder_specs,
    shard_transformer_specs,
    tokenize_prompt,
)


class ModelVariant(StrEnum):
    """Loadable components of the FLUX.2-dev pipeline."""

    TEXT_ENCODER = "TextEncoder"
    TRANSFORMER = "Transformer"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Load individual FLUX.2-dev components without instantiating Flux2Pipeline."""

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
            model="Flux2",
            variant=variant,
            group=ModelGroup.RED,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return Mistral3TextEncoderWrapper(load_text_encoder(dtype)).eval()
        if self._variant == ModelVariant.TRANSFORMER:
            return Flux2TransformerWrapper(load_transformer(dtype)).eval()
        if self._variant == ModelVariant.VAE:
            return Flux2VAEDecoderWrapper(load_vae(dtype)).eval()

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        Supported device counts: 1, 2, 4, 8, 32. All components run on the full
        device mesh — the PJRT runtime exposes every visible chip, so a program
        compiled for fewer chips fails with a device-count mismatch. The VAE has
        no shard spec, so it is simply replicated across the mesh.
        """
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return tensor → partition_spec dict for the active component."""
        if self._variant == ModelVariant.TEXT_ENCODER:
            return shard_text_encoder_specs(model.text_encoder)
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_transformer_specs(model.transformer)
        if self._variant == ModelVariant.VAE:
            return None
        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return synthetic inputs for the active component at {HEIGHT}x{WIDTH}."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            input_ids, attention_mask = tokenize_prompt(PROMPT)
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.TRANSFORMER:
            hidden_states = make_packed_latents(dtype)
            encoder_hidden_states = make_synthetic_prompt_embeds(dtype)
            # Flux2Pipeline passes scheduler timestep / 1000; model scales by 1000 internally.
            timestep = torch.tensor([500.0 / 1000.0], dtype=dtype)
            guidance = torch.tensor([GUIDANCE_SCALE], dtype=dtype)
            txt_ids = prepare_text_ids(1, MAX_SEQUENCE_LENGTH, dtype)
            img_ids = prepare_latent_image_ids(1, LATENT_GRID_H, LATENT_GRID_W, dtype)
            return [
                hidden_states,
                encoder_hidden_states,
                timestep,
                img_ids,
                txt_ids,
                guidance,
            ]

        if self._variant == ModelVariant.VAE:
            return [make_vae_decoder_input(dtype)]

        raise ValueError(f"Unknown variant: {self._variant}")
