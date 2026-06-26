# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image component loader.

Qwen-Image (``QwenImagePipeline``) is a text-to-image diffusion pipeline made of
independently-compilable components. Each ``ModelVariant`` selects one:

  - TextEncoder → Qwen2.5-VL text encoder            (8.29B)
  - Transformer → QwenImageTransformer2DModel (MMDiT) (20.43B, the per-step compute)
  - Vae         → AutoencoderKLQwenImage decoder      (~0.14B)

Following the FLUX / SD3 / HunyuanImage convention, ``load_model`` returns the
component as an ``nn.Module`` (wrapped for tensor-only I/O) so the tt-xla graph
tester can compile it without the host-side pipeline loop. The 40.9 GB
transformer does not fit one 32 GB chip, so it ships sharding hooks
(``get_mesh_config`` / ``load_shard_spec``) for tensor-parallel bringup.
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
    IMG_SEQ,
    MESH_NAMES,
    MESH_SHAPES,
    PACKED_HIDDEN_DIM,
    QWEN_VOCAB_SIZE,
    REPO_ID,
    TEXT_EMBED_DIM,
    TRANSFORMER_TEXT_SEQ,
    VAE_Z_DIM,
    LATENT_H,
    LATENT_W,
    QwenImageTransformerWrapper,
    QwenTextEncoderWrapper,
    VAEDecoderWrapper,
    load_text_encoder,
    load_transformer,
    load_vae,
    shard_text_encoder_specs,
    shard_transformer_specs,
)


class ModelVariant(StrEnum):
    """Loadable components of the Qwen-Image pipeline."""

    TEXT_ENCODER = "TextEncoder"
    TRANSFORMER = "Transformer"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Load individual Qwen-Image components without pulling the full pipeline."""

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=REPO_ID),
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
            if variant == ModelVariant.TEXT_ENCODER
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
            return QwenImageTransformerWrapper(load_transformer(dtype)).eval()
        if self._variant == ModelVariant.VAE:
            return VAEDecoderWrapper(load_vae(dtype)).eval()

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        The VAE fits a single chip, so any count maps to (1, 1).
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
            return shard_text_encoder_specs(model.encoder)
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_transformer_specs(model.transformer)
        return None

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a list of synthetic input tensors for the active component.

        TEXT_ENCODER → [input_ids (1,128) int64, attention_mask (1,128) int64]
        TRANSFORMER  → [hidden_states (1,6889,64), timestep (1,),
                        encoder_hidden_states (1,128,3584),
                        encoder_hidden_states_mask (1,128) int64]
        VAE          → [latents (1,16,1,166,166)]
        """
        dtype = dtype_override if dtype_override is not None else DTYPE
        torch.manual_seed(0)

        if self._variant == ModelVariant.TEXT_ENCODER:
            input_ids = torch.randint(
                0, QWEN_VOCAB_SIZE, (1, TRANSFORMER_TEXT_SEQ), dtype=torch.long
            )
            attention_mask = torch.ones(1, TRANSFORMER_TEXT_SEQ, dtype=torch.long)
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.TRANSFORMER:
            hidden_states = torch.randn(1, IMG_SEQ, PACKED_HIDDEN_DIM, dtype=dtype)
            timestep = torch.full((1,), 1.0, dtype=dtype)
            encoder_hidden_states = torch.randn(
                1, TRANSFORMER_TEXT_SEQ, TEXT_EMBED_DIM, dtype=dtype
            )
            encoder_hidden_states_mask = torch.ones(
                1, TRANSFORMER_TEXT_SEQ, dtype=torch.long
            )
            return [
                hidden_states,
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_mask,
            ]

        if self._variant == ModelVariant.VAE:
            latents = torch.randn(1, VAE_Z_DIM, 1, LATENT_H, LATENT_W, dtype=dtype)
            return [latents]

        raise ValueError(f"Unknown variant: {self._variant}")
