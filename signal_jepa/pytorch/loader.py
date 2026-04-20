# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SignalJEPA model loader implementation for EEG signal classification.

SignalJEPA is a Joint Embedding Predictive Architecture for cross-dataset
transfer learning on EEG (electroencephalography) data. It uses spatial
attention mechanisms to handle varying electrode configurations across datasets.
"""
import torch
from typing import Optional
from dataclasses import dataclass

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


@dataclass
class SignalJEPAConfig(ModelConfig):
    n_chans: int = 19
    n_times: int = 256
    model_class: str = "SignalJEPA_Contextual"


class ModelVariant(StrEnum):
    """Available SignalJEPA model variants."""

    CONTEXTUAL_PRETRAINED = "Contextual_Pretrained"
    PRETRAINED = "Pretrained"


class ModelLoader(ForgeModel):
    """SignalJEPA model loader for EEG signal classification."""

    _VARIANTS = {
        ModelVariant.CONTEXTUAL_PRETRAINED: SignalJEPAConfig(
            pretrained_model_name="braindecode/SignalJEPA-Contextual-pretrained",
            n_chans=19,
            n_times=256,
            model_class="SignalJEPA_Contextual",
        ),
        ModelVariant.PRETRAINED: SignalJEPAConfig(
            pretrained_model_name="braindecode/SignalJEPA-pretrained",
            n_chans=20,
            n_times=256,
            model_class="SignalJEPA",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTEXTUAL_PRETRAINED

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SignalJEPA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the SignalJEPA model.

        Returns:
            torch.nn.Module: The SignalJEPA model instance.
        """
        import braindecode.models as bd_models

        cfg = self._variant_config

        model_cls = getattr(bd_models, cfg.model_class)
        model = model_cls.from_pretrained(cfg.pretrained_model_name)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample EEG inputs for the model.

        Returns:
            torch.Tensor: Input tensor of shape (batch, n_channels, n_times).
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        inputs = torch.randn(1, cfg.n_chans, cfg.n_times, dtype=dtype)

        return inputs
