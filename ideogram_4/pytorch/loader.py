# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ideogram 4 component loader.

Ideogram 4 is a 9.3B single-stream DiT text-to-image model. Published weights
are FP8-only (``ideogram-ai/ideogram-4-fp8``). This loader materializes FP8
linear weights to bfloat16 at load time so tt-xla can compile them with TT
block formats (bfp_bf8 / bfp_bf4) via mixed-precision overrides.

Components:
  - Transformer_FP8_512 -> conditional Ideogram4Transformer DiT (9.3B)
  - TextEncoder         -> Qwen3-VL language tower (7.6B)
  - Vae                 -> AutoEncoder decoder (49.6M)

The transformer runs at 512x512 packed-sequence shapes matching the CPU
inference smoke test, and is tensor-parallel sharded; the VAE decoder fits on a
single chip. The text encoder produces the DiT's ``llm_features`` conditioning
and is unsharded, so it needs a chip that can hold ~15.2 GB of bf16 weights.
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
    MESH_NAMES,
    MESH_SHAPES,
    REPO_ID,
    Ideogram4TransformerWrapper,
    build_synthetic_transformer_inputs,
    build_text_encoder_inputs,
    build_vae_decoder_inputs,
    load_conditional_transformer,
    load_text_encoder,
    load_vae_decoder,
    shard_transformer_specs,
)


class ModelVariant(StrEnum):
    """Loadable Ideogram 4 components."""

    TRANSFORMER_FP8_512 = "Transformer_FP8_512"
    TEXT_ENCODER = "TextEncoder"
    VAE = "Vae"


class ModelLoader(ForgeModel):
    """Ideogram 4 conditional DiT loader (FP8 checkpoint → bf16 weights)."""

    _VARIANTS = {
        ModelVariant.TRANSFORMER_FP8_512: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.TEXT_ENCODER: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.VAE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER_FP8_512

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Ideogram4",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Return the conditional Ideogram4Transformer with FP8 weights → bf16."""
        dtype = dtype_override if dtype_override is not None else DTYPE
        if self._variant == ModelVariant.TEXT_ENCODER:
            return load_text_encoder(dtype=dtype)
        if self._variant == ModelVariant.VAE:
            return load_vae_decoder(dtype=dtype)
        transformer = load_conditional_transformer(dtype=dtype)
        return Ideogram4TransformerWrapper(transformer).eval()

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Synthetic packed-sequence inputs for 512x512 resolution."""
        dtype = dtype_override if dtype_override is not None else DTYPE
        if self._variant == ModelVariant.TEXT_ENCODER:
            return build_text_encoder_inputs(dtype=dtype)
        if self._variant == ModelVariant.VAE:
            return build_vae_decoder_inputs(dtype=dtype)
        return build_synthetic_transformer_inputs(batch_size=batch_size, dtype=dtype)

    def unpack_forward_output(self, output):
        """Return the component's output tensor.

        The DiT returns a velocity prediction, the text encoder the packed
        llm_features and the VAE decoder an image; all three are bare tensors.
        """
        if isinstance(output, tuple):
            return output[0]
        return output

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        The transformer is sharded tensor-parallel along the "model" axis. The
        VAE decoder is 49.6M params and fits on a single chip, so it maps to
        (1, 1) for any device count. The text encoder is also unsharded -- see
        load_shard_spec.
        """
        if self._variant in (ModelVariant.TEXT_ENCODER, ModelVariant.VAE):
            return (1, 1), MESH_NAMES
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return tensor -> partition_spec dict (Option A, Megatron column->row).

        Expects the wrapper returned by load_model(); shards the inner transformer.
        The VAE decoder and the text encoder are unsharded. Qwen3-VL is in fact
        cleanly shardable (32 q / 8 kv heads and a 12288 MLP all divide 2, 4 and
        8), unlike this DiT's fused 18-head attention -- but no spec is declared
        here because it could not be validated on the single chip this component
        was brought up on.
        """
        if self._variant in (ModelVariant.TEXT_ENCODER, ModelVariant.VAE):
            return None
        return shard_transformer_specs(model.transformer)
