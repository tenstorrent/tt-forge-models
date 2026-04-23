# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ECGFounder (PKUDigitalHealth/ECGFounder) model loader implementation.

ECGFounder is a 1-D CNN foundation model pretrained on 10M+ ECG recordings for
electrocardiogram signal classification. Two checkpoint variants are published:
a 12-lead model and a 1-lead model, both operating on 10-second recordings at
500 Hz sampling rate (5000 samples per lead).
"""

from dataclasses import dataclass
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


class ModelVariant(StrEnum):
    """Available ECGFounder model variants."""

    LEAD_12 = "12_lead"
    LEAD_1 = "1_lead"


@dataclass
class ECGFounderConfig(ModelConfig):
    """Configuration for an ECGFounder variant."""

    checkpoint_filename: str = ""
    in_channels: int = 12


class ModelLoader(ForgeModel):
    """ECGFounder model loader implementation (PyTorch)."""

    # ECGFounder was pretrained to predict 150 cardiovascular conditions.
    N_CLASSES = 150
    # 10 seconds @ 500 Hz.
    INPUT_LENGTH = 5000

    _VARIANTS = {
        ModelVariant.LEAD_12: ECGFounderConfig(
            pretrained_model_name="PKUDigitalHealth/ECGFounder",
            checkpoint_filename="12_lead_ECGFounder.pth",
            in_channels=12,
        ),
        ModelVariant.LEAD_1: ECGFounderConfig(
            pretrained_model_name="PKUDigitalHealth/ECGFounder",
            checkpoint_filename="1_lead_ECGFounder.pth",
            in_channels=1,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LEAD_12

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ECGFounder",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import os

        import torch

        from .src.model import Net1D

        config = self._variant_config

        model = Net1D(
            in_channels=config.in_channels,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            n_classes=self.N_CLASSES,
            use_bn=False,
            use_do=False,
        )

        if not os.environ.get("TT_RANDOM_WEIGHTS"):
            from huggingface_hub import hf_hub_download

            checkpoint_path = hf_hub_download(
                repo_id=config.pretrained_model_name,
                filename=config.checkpoint_filename,
            )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            state_dict = (
                checkpoint["state_dict"]
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                else checkpoint
            )
            model.load_state_dict(state_dict, strict=False)

        model.eval()
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        import torch

        in_channels = self._variant_config.in_channels
        inputs = torch.randn(batch_size, in_channels, self.INPUT_LENGTH)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
