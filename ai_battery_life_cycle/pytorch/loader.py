# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NeerajCodz/aiBatteryLifeCycle model loader implementation.

aiBatteryLifeCycle is a tabular regression model for predicting
State-of-Health (SOH) of lithium-ion batteries from a window of
physics-informed cycle features, trained on the NASA PCoE Battery Dataset.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from huggingface_hub import hf_hub_download

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model import BidirectionalLSTM, GRUModel, VanillaLSTM


@dataclass
class BatteryLifeCycleConfig(ModelConfig):
    """Configuration specific to aiBatteryLifeCycle deep learning variants."""

    weights_file: str = ""
    architecture: str = "vanilla_lstm"
    input_size: int = 18
    hidden_size: int = 128
    num_layers: int = 2
    sequence_length: int = 32


class ModelVariant(StrEnum):
    """Available aiBatteryLifeCycle PyTorch model variants."""

    VANILLA_LSTM = "vanilla_lstm"
    BIDIRECTIONAL_LSTM = "bidirectional_lstm"
    GRU = "gru"


class ModelLoader(ForgeModel):
    """NeerajCodz/aiBatteryLifeCycle model loader for battery SOH regression."""

    _REPO_ID = "NeerajCodz/aiBatteryLifeCycle"

    _VARIANTS = {
        ModelVariant.VANILLA_LSTM: BatteryLifeCycleConfig(
            pretrained_model_name=_REPO_ID,
            weights_file="v3/models/deep/vanilla_lstm.pt",
            architecture="vanilla_lstm",
        ),
        ModelVariant.BIDIRECTIONAL_LSTM: BatteryLifeCycleConfig(
            pretrained_model_name=_REPO_ID,
            weights_file="v3/models/deep/bidirectional_lstm.pt",
            architecture="bidirectional_lstm",
        ),
        ModelVariant.GRU: BatteryLifeCycleConfig(
            pretrained_model_name=_REPO_ID,
            weights_file="v3/models/deep/gru.pt",
            architecture="gru",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VANILLA_LSTM

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="aiBatteryLifeCycle",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_model(self) -> torch.nn.Module:
        config = self._variant_config
        kwargs = {
            "input_size": config.input_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
        }
        if config.architecture == "vanilla_lstm":
            return VanillaLSTM(**kwargs)
        if config.architecture == "bidirectional_lstm":
            return BidirectionalLSTM(**kwargs)
        if config.architecture == "gru":
            return GRUModel(**kwargs)
        raise ValueError(f"Unsupported architecture: {config.architecture}")

    def load_model(self, *, dtype_override=None, **kwargs):
        """Build the architecture and load pretrained weights from the HF Hub."""
        config = self._variant_config
        model = self._build_model()

        weights_path = hf_hub_download(
            repo_id=config.pretrained_model_name, filename=config.weights_file
        )
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return a random cycle-feature window of shape (batch, seq, features)."""
        config = self._variant_config
        inputs = torch.randn(batch_size, config.sequence_length, config.input_size)
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)
        return inputs
