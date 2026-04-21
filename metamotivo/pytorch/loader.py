# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Meta Motivo model loader implementation for humanoid action prediction.
"""

from __future__ import annotations

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


class MetaMotivoActorWrapper(torch.nn.Module):
    """Wraps FBcprModel to expose the actor policy forward pass.

    FBcprModel.act(obs, z, mean=True) returns the deterministic action; wrap it
    so the standard forward(obs, z) signature drives inference.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.model.act(obs, z, mean=True)


class ModelVariant(StrEnum):
    """Available Meta Motivo model variants."""

    S_1 = "S-1"


class ModelLoader(ForgeModel):
    """Meta Motivo model loader for zero-shot humanoid whole-body control."""

    _VARIANTS = {
        ModelVariant.S_1: ModelConfig(
            pretrained_model_name="facebook/metamotivo-S-1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.S_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="metamotivo",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from metamotivo.fb_cpr.huggingface import FBcprModel

        model = FBcprModel.from_pretrained(self._variant_config.pretrained_model_name)
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)
        model.eval()
        self.model = model
        return MetaMotivoActorWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32
        obs = torch.randn(batch_size, self.model.cfg.obs_dim, dtype=dtype)
        z = self.model.sample_z(batch_size).to(dtype=dtype)

        return {"obs": obs, "z": z}
