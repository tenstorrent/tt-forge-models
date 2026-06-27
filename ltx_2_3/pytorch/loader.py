# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 (Lightricks) audio-video generation component loader.

Each variant corresponds to one independently loadable component of the
diffusion pipeline. The HuggingFace repo ships a single monolithic checkpoint;
the text encoder/tokenizer live externally, so only the on-checkpoint
components are exposed here.

  - Transformer -> LTX2VideoTransformer3DModel (DiT denoiser)   params=~21B
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
    REPO_ID,
    DTYPE,
    MESH_NAMES,
    MESH_SHAPES,
    NUM_ATTENTION_HEADS,
    LTX2TransformerWrapper,
    load_transformer,
    load_transformer_inputs,
    shard_transformer_specs,
)


class ModelVariant(StrEnum):
    """Loadable components of the LTX-2.3 pipeline."""

    TRANSFORMER = "Transformer"


class ModelLoader(ForgeModel):
    """Load individual LTX-2.3 components without pulling the full pipeline."""

    _VARIANTS = {
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LTX-2.3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module.

        Returns:
            TRANSFORMER -> LTX2TransformerWrapper (video noise prediction)
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TRANSFORMER:
            return LTX2TransformerWrapper(load_transformer(dtype)).eval()

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        The ~21B denoiser (~44 GB bf16) exceeds a single 32 GB Blackhole chip, so
        a tensor-parallel "model" axis is the baseline. num_attention_heads (32)
        is divisible by every supported model-axis size.
        """
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        mesh_shape = MESH_SHAPES[num_devices]
        if NUM_ATTENTION_HEADS % mesh_shape[1] != 0:
            raise ValueError(
                f"num_attention_heads={NUM_ATTENTION_HEADS} not divisible by "
                f"model-axis size {mesh_shape[1]}."
            )
        return mesh_shape, MESH_NAMES

    def load_shard_spec(self, model):
        """Return tensor -> partition_spec dict for the active component.

        Expects the same wrapper returned by load_model(); specs are derived
        from the wrapped `.transformer`.
        """
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_transformer_specs(model.transformer)
        return None

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a list of synthetic input tensors for the active component.

        TRANSFORMER -> [hidden_states, audio_hidden_states, encoder_hidden_states,
                        audio_encoder_hidden_states, timestep, sigma]
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TRANSFORMER:
            return load_transformer_inputs(dtype)

        raise ValueError(f"Unknown variant: {self._variant}")
