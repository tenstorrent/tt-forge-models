# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image (text-to-image MMDiT) component loader.

Each variant corresponds to one independently loadable / compilable component:
  - TextEncoder → Qwen2.5-VL text decoder (text_encoder)        params ~8.3B
  - Transformer → QwenImageTransformerWrapper (MMDiT, 60 blocks) params ~20.4B
  - Vae         → VAEDecoderWrapper (AutoencoderKLQwenImage)     params ~0.25B
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
    IMG_SEQ_LEN,
    LATENT_H,
    LATENT_W,
    MESH_NAMES,
    MESH_SHAPES,
    QWEN_VOCAB_SIZE,
    TEXT_EMBED_DIM,
    TEXT_SEQ_LEN,
    TRANSFORMER_IN_CHANNELS,
    Z_DIM,
    QwenImageTransformerWrapper,
    VAEDecoderWrapper,
    load_text_encoder,
    load_transformer,
    load_vae,
    shard_text_encoder_specs,
    shard_transformer_specs,
)

_REPO_ID = "Qwen/Qwen-Image"

# Qwen2.5-VL tokenizer max length used by the pipeline (max_sequence_length=512);
# the component test uses a representative shorter prompt.
TEXT_TOKEN_MAX_LEN = TEXT_SEQ_LEN


class ModelVariant(StrEnum):
    """Loadable components of the Qwen-Image pipeline."""

    TEXT_ENCODER = "TextEncoder"
    TRANSFORMER = "Transformer"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Load individual Qwen-Image components without pulling the full pipeline."""

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=_REPO_ID),
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
            if variant == ModelVariant.TEXT_ENCODER
            else ModelTask.MM_IMAGE_TTT
        )
        return ModelInfo(
            model="QwenImage",
            variant=variant,
            group=ModelGroup.RED,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module.

        Returns:
            TEXT_ENCODER → Qwen2.5-VL text decoder
            TRANSFORMER  → QwenImageTransformerWrapper
            VAE          → VAEDecoderWrapper (decoder-only, returns plain tensor)
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return load_text_encoder(dtype)
        if self._variant == ModelVariant.TRANSFORMER:
            return QwenImageTransformerWrapper(load_transformer(dtype)).eval()
        if self._variant == ModelVariant.VAE:
            return VAEDecoderWrapper(load_vae(dtype)).eval()

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

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
        """Return tensor → partition_spec dict for the active component."""
        if self._variant == ModelVariant.TEXT_ENCODER:
            return shard_text_encoder_specs(model)
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_transformer_specs(model.transformer)
        return None

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a list of synthetic input tensors for the active component.

        TEXT_ENCODER → [input_ids (1,256) int64, attention_mask (1,256) int64]
        TRANSFORMER  → [hidden_states (1,6889,64), timestep (1,),
                        encoder_hidden_states (1,256,3584),
                        encoder_hidden_states_mask (1,256) int64]
        VAE          → [z (1,16,1,166,166)]   (3D causal VAE, T=1)
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            input_ids = torch.randint(
                0, QWEN_VOCAB_SIZE, (1, TEXT_TOKEN_MAX_LEN), dtype=torch.long
            )
            attention_mask = torch.ones(1, TEXT_TOKEN_MAX_LEN, dtype=torch.long)
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.TRANSFORMER:
            hidden_states = torch.randn(1, IMG_SEQ_LEN, TRANSFORMER_IN_CHANNELS, dtype=dtype)
            # timestep is already divided by 1000 at the pipeline call site
            timestep = torch.tensor([1.0], dtype=dtype)
            encoder_hidden_states = torch.randn(1, TEXT_SEQ_LEN, TEXT_EMBED_DIM, dtype=dtype)
            encoder_hidden_states_mask = torch.ones(1, TEXT_SEQ_LEN, dtype=torch.long)
            return [
                hidden_states,
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_mask,
            ]

        if self._variant == ModelVariant.VAE:
            z = torch.randn(1, Z_DIM, 1, LATENT_H, LATENT_W, dtype=dtype)
            return [z]

        raise ValueError(f"Unknown variant: {self._variant}")
