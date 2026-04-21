# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Parakeet TDT HF model loader implementation for automatic speech recognition.

Uses NeMo ASR to load the model since the HuggingFace transformers integration
for parakeet_tdt model type is not yet available in a released version.
"""

import torch
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
    """Available Parakeet TDT HF model variants."""

    TDT_0_6B_V3_HF = "TDT 0.6B v3 HF"


class ModelLoader(ForgeModel):
    """Parakeet TDT HF model loader implementation for automatic speech recognition."""

    _VARIANTS = {
        ModelVariant.TDT_0_6B_V3_HF: ModelConfig(
            pretrained_model_name="nvidia/parakeet-tdt-0.6b-v3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TDT_0_6B_V3_HF

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Parakeet_TDT_HF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import nemo.collections.asr as nemo_asr

        model = nemo_asr.models.ASRModel.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        input_signal = torch.tensor(audio_array).unsqueeze(0)
        input_signal_length = torch.tensor([len(audio_array)])

        if dtype_override is not None:
            input_signal = input_signal.to(dtype_override)

        return {
            "input_signal": input_signal,
            "input_signal_length": input_signal_length,
        }
