# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ConvLSTM Autoencoder (JensLundsgaard/control-2026-03-04) model loader.
"""
import torch
from typing import Optional

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
from .src.model import ConvLSTMAutoencoder


class ModelVariant(StrEnum):
    """Available ConvLSTM Autoencoder model variants."""

    CONTROL_2026_03_04 = "control-2026-03-04"


class ModelLoader(ForgeModel):
    """ConvLSTM Autoencoder model loader for video reconstruction."""

    _VARIANTS = {
        ModelVariant.CONTROL_2026_03_04: ModelConfig(
            pretrained_model_name="JensLundsgaard/control-2026-03-04",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROL_2026_03_04

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ConvLSTMAutoencoder",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the ConvLSTM Autoencoder model from HuggingFace Hub.

        Returns:
            torch.nn.Module: The ConvLSTMAutoencoder model instance.
        """
        cfg = self._variant_config

        model = ConvLSTMAutoencoder.from_pretrained(cfg.pretrained_model_name)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample video inputs for the ConvLSTM Autoencoder model.

        Returns:
            torch.Tensor: Input tensor of shape (B, T, 1, 128, 128).
        """
        seq_len = 50
        input_channels = 1
        image_size = 128

        torch.manual_seed(42)
        inputs = torch.rand(batch_size, seq_len, input_channels, image_size, image_size)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
