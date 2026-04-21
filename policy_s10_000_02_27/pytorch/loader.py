# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Model loader for daphne-cornelisse/policy_S10_000_02_27.

A feed-forward late-fusion actor-critic policy trained with self-play in the
gpudrive autonomous driving simulator (arXiv:2502.14706). Weights are published
via `PyTorchModelHubMixin` without a custom model class in the HF repo, so we
instantiate a standalone mirror of the gpudrive `NeuralNet` architecture and
load the published safetensors.
"""
import torch
from typing import Optional

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
from .src.model import NeuralNet


class ModelVariant(StrEnum):
    """Available model variants."""

    POLICY_S10_000_02_27 = "policy_S10_000_02_27"


class ModelLoader(ForgeModel):
    """Loader for the gpudrive FFN policy (policy_S10_000_02_27)."""

    _VARIANTS = {
        ModelVariant.POLICY_S10_000_02_27: ModelConfig(
            pretrained_model_name="daphne-cornelisse/policy_S10_000_02_27",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.POLICY_S10_000_02_27

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GPUDrivePolicy",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the gpudrive FFN policy network with pretrained weights."""
        repo_id = self._variant_config.pretrained_model_name
        model = NeuralNet.from_pretrained(repo_id)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return a flattened observation tensor of shape (batch_size, obs_dim)."""
        obs_dim = 2984

        torch.manual_seed(42)
        obs = torch.randn(batch_size, obs_dim, dtype=torch.float32)

        if dtype_override is not None:
            obs = obs.to(dtype_override)

        return obs
