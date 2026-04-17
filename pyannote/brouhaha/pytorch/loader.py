# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pyannote brouhaha model loader implementation.

Loads the brouhaha model for joint voice activity detection, speech-to-noise
ratio (SNR), and C50 room acoustics estimation.
"""

import os

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
    """Available Pyannote brouhaha model variants."""

    BROUHAHA = "Brouhaha"


class ModelLoader(ForgeModel):
    """Pyannote brouhaha model loader implementation.

    Loads the brouhaha model for joint VAD, SNR, and C50 estimation.
    """

    _VARIANTS = {
        ModelVariant.BROUHAHA: ModelConfig(
            pretrained_model_name="pyannote/brouhaha",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BROUHAHA

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
        """Load the Pyannote brouhaha model.

        Tries HuggingFace first (requires gated access); falls back to
        constructing the architecture locally with random weights.
        """
        from pyannote.audio import Model

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        token = kwargs.pop("token", None) or os.environ.get("HF_TOKEN")
        if token:
            model_kwargs["use_auth_token"] = token
        model_kwargs |= kwargs

        try:
            self._model = Model.from_pretrained(
                self._variant_config.pretrained_model_name, **model_kwargs
            )
        except Exception:
            self._model = self._build_model_from_scratch()

        self._model.eval()
        if dtype_override is not None:
            self._model.to(dtype_override)
        return self._model

    @staticmethod
    def _build_model_from_scratch():
        from pyannote.audio.models.segmentation import SSeRiouSS
        from pyannote.audio.core.task import (
            Problem,
            Resolution,
            Specifications,
        )

        model = SSeRiouSS(
            wav2vec="WAVLM_BASE",
            sample_rate=16000,
            num_channels=1,
        )
        model.specifications = Specifications(
            problem=Problem.MULTI_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=10.0,
            classes=["vad", "snr_db", "c50_db"],
        )
        model.build()
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the brouhaha model.

        Generates a 10-second mono audio waveform at 16kHz as expected
        by the model: shape (batch_size, num_channels, num_samples) = (1, 1, 160000).
        """
        dtype = dtype_override or torch.float32
        # 10 seconds of mono audio at 16kHz
        waveform = torch.randn(1, 1, 160000, dtype=dtype)
        return [waveform]
