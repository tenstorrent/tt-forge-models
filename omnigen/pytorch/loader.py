# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OmniGen component loader.

Each variant corresponds to one independently loadable component:
  - Transformer → OmniGenTransformerWrapper (DiT with built-in text embedding)
  - Vae         → VAEDecoderWrapper (decoder-only — final latent → image step)
  - VaeEncoder  → VAEEncoderWrapper (encoder-only — image conditioning path)

OmniGen embeds text tokens inside the transformer (LLaMA-style
`embed_tokens`), so there is no separate text encoder variant.
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
    OmniGenTransformerWrapper,
    VAEDecoderWrapper,
    VAEEncoderWrapper,
    load_transformer,
    load_transformer_inputs,
    load_vae,
    load_vae_encoder_inputs,
    load_vae_decoder_inputs,
    shard_transformer_specs,
    shard_vae_encoder_specs,
    shard_vae_specs,
)


class ModelVariant(StrEnum):
    """Loadable components of the OmniGen pipeline."""

    TRANSFORMER = "Transformer"
    VAE_DECODER = "Vae"
    VAE_ENCODER = "VaeEncoder"


class ModelLoader(ForgeModel):
    """Load individual OmniGen components without pulling the full pipeline."""

    _VARIANTS = {
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.VAE_DECODER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.VAE_ENCODER: ModelConfig(pretrained_model_name=REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="OmniGen",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module.

        Returns:
            TRANSFORMER → OmniGenTransformerWrapper
            VAE_DECODER         → VAEDecoderWrapper (decoder-only, returns plain tensor)
            VAE_ENCODER → VAEEncoderWrapper (encoder + quant_conv, returns
                          posterior mean * scaling_factor)
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TRANSFORMER:
            return OmniGenTransformerWrapper(load_transformer(dtype)).eval()
        if self._variant == ModelVariant.VAE_DECODER:
            return VAEDecoderWrapper(load_vae(dtype)).eval()
        if self._variant == ModelVariant.VAE_ENCODER:
            return VAEEncoderWrapper(load_vae(dtype)).eval()

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        Supported device counts: 1, 2, 4, 8, 32. Both components are sharded
        so that the full pipeline can run on a single mesh.
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
          TRANSFORMER → OmniGenTransformerWrapper (specs from .transformer)
          VAE_DECODER         → VAEDecoderWrapper (specs from .vae, decoder path)
          VAE_ENCODER → VAEEncoderWrapper (specs from .vae, encoder path)
        """
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_transformer_specs(model.transformer)
        if self._variant == ModelVariant.VAE_DECODER:
            return shard_vae_specs(model.vae)
        if self._variant == ModelVariant.VAE_ENCODER:
            return shard_vae_encoder_specs(model.vae)
        return None

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a list of synthetic input tensors for the active component.

        TRANSFORMER → [hidden_states, timestep, input_ids,
                       attention_mask, position_ids]
        VAE_DECODER         → [z (1, 4, 16, 16) bfloat16]
        VAE_ENCODER → [x (1, 3, 128, 128) bfloat16]
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TRANSFORMER:
            return load_transformer_inputs(dtype)
        if self._variant == ModelVariant.VAE_DECODER:
            return load_vae_decoder_inputs(dtype)
        if self._variant == ModelVariant.VAE_ENCODER:
            return load_vae_encoder_inputs(dtype)

        raise ValueError(f"Unknown variant: {self._variant}")
