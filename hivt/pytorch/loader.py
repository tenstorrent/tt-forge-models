# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HiVT model loader for vehicle trajectory prediction.

Reference: https://github.com/ZikangZhou/HiVT
"HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction" (CVPR 2022).
"""

import torch
from typing import Optional
from dataclasses import dataclass

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from .src.model import HiVT


@dataclass
class HiVTConfig(ModelConfig):
    """Configuration specific to HiVT variants."""

    hidden_dim: int = 64
    num_heads: int = 4
    local_layers: int = 2
    global_layers: int = 2
    history_steps: int = 20
    future_steps: int = 30
    num_modes: int = 6
    num_agents: int = 10


class ModelVariant(StrEnum):
    """Available HiVT model variants."""

    HIVT_64 = "HiVT_64"
    HIVT_128 = "HiVT_128"


class ModelLoader(ForgeModel):
    """HiVT model loader for trajectory prediction."""

    _VARIANTS = {
        ModelVariant.HIVT_64: HiVTConfig(
            pretrained_model_name="hivt_64",
            hidden_dim=64,
            num_heads=4,
            local_layers=2,
            global_layers=2,
            history_steps=20,
            future_steps=30,
            num_modes=6,
            num_agents=10,
        ),
        ModelVariant.HIVT_128: HiVTConfig(
            pretrained_model_name="hivt_128",
            hidden_dim=128,
            num_heads=8,
            local_layers=3,
            global_layers=3,
            history_steps=20,
            future_steps=30,
            num_modes=6,
            num_agents=10,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HIVT_64

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="HiVT",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the HiVT model.

        Args:
            dtype_override: Optional torch.dtype override.

        Returns:
            torch.nn.Module: HiVT model instance.
        """
        cfg = self._variant_config
        model = HiVT(
            in_dim=2,
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            local_layers=cfg.local_layers,
            global_layers=cfg.global_layers,
            future_steps=cfg.future_steps,
            num_modes=cfg.num_modes,
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Generate sample inputs for HiVT.

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).

        Returns:
            tuple: (agent_history,) — (B, N, T, 2) displacement history
        """
        cfg = self._variant_config
        dtype = dtype_override if dtype_override is not None else torch.float32

        agent_history = torch.randn(
            batch_size, cfg.num_agents, cfg.history_steps, 2, dtype=dtype
        )
        return (agent_history,)
