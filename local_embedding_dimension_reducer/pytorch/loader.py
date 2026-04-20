# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DialOnce local-EmbeddingDimensionReducer loader.

A small two-layer MLP (Linear(1536, 1536) -> ReLU -> Linear(1536, 2)) shipped
with PyTorchModelHubMixin. The HF repo does not include the source module, so
the architecture is reconstructed from the safetensors tensor names/shapes and
``config.json`` fields (``input_dim``, ``hidden_size``, ``output_dim``).
"""
from dataclasses import dataclass
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


@dataclass
class EmbeddingDimensionReducerConfig(ModelConfig):
    input_dim: int = 1536


class ModelVariant(StrEnum):
    """Available model variants."""

    DIALONCE_LOCAL = "dialonce-local"


class ModelLoader(ForgeModel):
    """Loader for DialOnce/local-EmbeddingDimensionReducer."""

    _VARIANTS = {
        ModelVariant.DIALONCE_LOCAL: EmbeddingDimensionReducerConfig(
            pretrained_model_name="DialOnce/local-EmbeddingDimensionReducer",
            input_dim=1536,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DIALONCE_LOCAL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="EmbeddingDimensionReducer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from .src.model import EmbeddingDimensionReducer

        cfg = self._variant_config

        model = EmbeddingDimensionReducer.from_pretrained(cfg.pretrained_model_name)

        # Safetensors ships as float64; cast to float32 for consistency with
        # repo-wide default inputs unless the caller overrides.
        target_dtype = dtype_override if dtype_override is not None else torch.float32
        model = model.to(target_dtype)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        x = torch.randn(1, cfg.input_dim, dtype=dtype)

        return {"x": x}
