# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Decision Transformer model loader implementation for offline RL action prediction.
"""
import torch
from transformers import DecisionTransformerModel, AutoConfig
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Decision Transformer model variants."""

    GYM_HOPPER_MEDIUM = "gym-hopper-medium"


class ModelLoader(ForgeModel):
    """Decision Transformer model loader implementation for action prediction tasks."""

    _VARIANTS = {
        ModelVariant.GYM_HOPPER_MEDIUM: ModelConfig(
            pretrained_model_name="edbeeching/decision-transformer-gym-hopper-medium",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GYM_HOPPER_MEDIUM

    # Sequence length for the sampled trajectory input window
    sequence_length = 20

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Decision Transformer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = DecisionTransformerModel.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.config is None:
            self.config = AutoConfig.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        state_dim = self.config.state_dim
        act_dim = self.config.act_dim
        seq_len = self.sequence_length

        dtype = dtype_override if dtype_override is not None else torch.float32

        states = torch.randn(batch_size, seq_len, state_dim, dtype=dtype)
        actions = torch.randn(batch_size, seq_len, act_dim, dtype=dtype)
        rewards = torch.randn(batch_size, seq_len, dtype=dtype)
        returns_to_go = torch.randn(batch_size, seq_len, 1, dtype=dtype)
        timesteps = (
            torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        )
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "returns_to_go": returns_to_go,
            "timesteps": timesteps,
            "attention_mask": attention_mask,
        }

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
