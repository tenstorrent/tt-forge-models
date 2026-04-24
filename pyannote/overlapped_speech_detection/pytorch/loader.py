# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pyannote overlapped speech detection model loader implementation.

Loads the overlapped-speech-detection pipeline and extracts its segmentation
model for testing, as this is the primary neural network component.
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
    """Available Pyannote overlapped speech detection model variants."""

    TEZUESH_OSD = "Tezuesh_OSD"


class ModelLoader(ForgeModel):
    """Pyannote overlapped speech detection model loader implementation.

    Loads the overlapped-speech-detection pipeline and extracts its
    segmentation model for testing.
    """

    _VARIANTS = {
        ModelVariant.TEZUESH_OSD: ModelConfig(
            pretrained_model_name="tezuesh/overlapped-speech-detection",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TEZUESH_OSD

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
        """Load the Pyannote overlapped speech detection pipeline's segmentation model.

        Requires a HuggingFace token with access to the gated model.
        Set the HF_TOKEN environment variable or pass token as a kwarg.
        """
        import torchaudio

        # pyannote.audio 3.x uses torchaudio APIs removed in torchaudio 2.9+;
        # patch them back in before importing so the import succeeds
        if not hasattr(torchaudio, "AudioMetaData"):
            from typing import NamedTuple

            class AudioMetaData(NamedTuple):
                sample_rate: int
                num_frames: int
                num_channels: int
                bits_per_sample: int
                encoding: str

            torchaudio.AudioMetaData = AudioMetaData

        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]

        if not hasattr(torchaudio, "info"):
            torchaudio.info = lambda path, backend=None: torchaudio.AudioMetaData(
                sample_rate=16000,
                num_frames=160000,
                num_channels=1,
                bits_per_sample=16,
                encoding="PCM_S",
            )

        # pyannote.audio 3.x passes use_auth_token to hf_hub_download which
        # was removed in huggingface_hub 1.x; patch to convert it to token
        import huggingface_hub as _hf_hub

        _orig_hf_hub_download = _hf_hub.hf_hub_download

        def _patched_hf_hub_download(*args, use_auth_token=None, token=None, **kwargs):
            if use_auth_token is not None and token is None:
                token = use_auth_token
            return _orig_hf_hub_download(*args, token=token, **kwargs)

        _hf_hub.hf_hub_download = _patched_hf_hub_download

        from pyannote.audio import Pipeline

        pipeline_kwargs = {}
        token = kwargs.pop("token", None) or os.environ.get("HF_TOKEN")
        if token:
            # pyannote.audio 3.x uses use_auth_token instead of token
            pipeline_kwargs["use_auth_token"] = token

        pipeline = Pipeline.from_pretrained(
            self._variant_config.pretrained_model_name, **pipeline_kwargs
        )

        # Extract the segmentation model from the pipeline
        self._model = pipeline._segmentation.model
        self._model.eval()
        if dtype_override is not None:
            self._model.to(dtype_override)
        return self._model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the OSD segmentation model.

        Generates a 10-second mono audio waveform at 16kHz as expected
        by the model: shape (batch_size, num_channels, num_samples) = (1, 1, 160000).
        """
        dtype = dtype_override or torch.float32
        # 10 seconds of mono audio at 16kHz
        waveform = torch.randn(1, 1, 160000, dtype=dtype)
        return [waveform]
