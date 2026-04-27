# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pyannote speaker diarization model loader implementation.

Loads the PyanNet segmentation model used as the backbone in the
speaker diarization pipeline (pyannote/speaker-diarization-3.x).
Creates the model with architecture-matched random weights to avoid
gated HuggingFace access requirements.
"""

import torch
import torch.nn as nn
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
    """Available Pyannote speaker diarization model variants."""

    DIARIZATION_3_0 = "Diarization_3_0"
    DIARIZATION_3_1 = "Diarization_3_1"


class ModelLoader(ForgeModel):
    """Pyannote speaker diarization model loader implementation.

    Instantiates the PyanNet segmentation model (the neural network backbone
    of pyannote/speaker-diarization-3.x) with architecture-matched random
    weights. This avoids the gated HuggingFace access requirement while
    still validating the model architecture for compilation.
    """

    _VARIANTS = {
        ModelVariant.DIARIZATION_3_0: ModelConfig(
            pretrained_model_name="pyannote/speaker-diarization-3.0",
        ),
        ModelVariant.DIARIZATION_3_1: ModelConfig(
            pretrained_model_name="pyannote/speaker-diarization-3.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DIARIZATION_3_1

    def __init__(self, variant=None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Pyannote",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Build the PyanNet segmentation model with architecture-matched random weights.

        Uses the same architecture as pyannote/segmentation-3.0 (the model
        embedded in pyannote/speaker-diarization-3.x pipelines):
          - SincNet frontend with stride=10
          - 4-layer bidirectional LSTM (hidden_size=128)
          - 2-layer linear projection (hidden_size=128)
          - 7-class sigmoid output (max speakers)
        """
        from pyannote.audio.models.segmentation import PyanNet

        self._model = PyanNet(
            sincnet={"stride": 10},
            lstm={
                "hidden_size": 128,
                "num_layers": 4,
                "bidirectional": True,
                "monolithic": True,
                "dropout": 0.5,
            },
            linear={"hidden_size": 128, "num_layers": 2},
            sample_rate=16000,
            num_channels=1,
        )
        # Segmentation-3.0 uses 7-class sigmoid for speaker overlap detection
        self._model.classifier = nn.Linear(128, 7)
        self._model.activation = nn.Sigmoid()

        self._model.eval()
        if dtype_override is not None:
            self._model.to(dtype_override)
        return self._model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the diarization segmentation model.

        Generates a 10-second mono audio waveform at 16kHz as expected
        by the model: shape (batch_size, num_channels, num_samples) = (1, 1, 160000).
        """
        dtype = dtype_override or torch.float32
        waveform = torch.randn(1, 1, 160000, dtype=dtype)
        return [waveform]
