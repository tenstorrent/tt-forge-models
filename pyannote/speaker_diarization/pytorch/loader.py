# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pyannote speaker diarization model loader implementation.

Loads the speaker diarization pipeline and extracts its segmentation
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
    """Available Pyannote speaker diarization model variants."""

    DIARIZATION_3_0 = "Diarization_3_0"
    DIARIZATION_3_1 = "Diarization_3_1"
    DIARIZATION_COMMUNITY_1 = "Diarization_Community_1"
    TEZUESH_DIARIZATION = "Tezuesh_Diarization"
    FATYMATARIQ_DIARIZATION_3_1 = "Fatymatariq_Diarization_3_1"


class ModelLoader(ForgeModel):
    """Pyannote speaker diarization model loader implementation.

    Loads the speaker diarization pipeline and extracts its
    segmentation model for testing.
    """

    _VARIANTS = {
        ModelVariant.DIARIZATION_3_0: ModelConfig(
            pretrained_model_name="pyannote/speaker-diarization-3.0",
        ),
        ModelVariant.DIARIZATION_3_1: ModelConfig(
            pretrained_model_name="pyannote/speaker-diarization-3.1",
        ),
        ModelVariant.DIARIZATION_COMMUNITY_1: ModelConfig(
            pretrained_model_name="pyannote/speaker-diarization-community-1",
        ),
        ModelVariant.TEZUESH_DIARIZATION: ModelConfig(
            pretrained_model_name="tezuesh/diarization",
        ),
        ModelVariant.FATYMATARIQ_DIARIZATION_3_1: ModelConfig(
            pretrained_model_name="fatymatariq/speaker-diarization-3.1",
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
        """Load the Pyannote speaker diarization segmentation model.

        Downloads the pipeline config to locate the segmentation sub-model,
        then loads only that model directly to avoid gated PLDA dependencies.
        """
        import yaml
        from huggingface_hub import hf_hub_download
        from pyannote.audio import Model

        token = kwargs.pop("token", None) or os.environ.get("HF_TOKEN")
        hub_kwargs = {}
        if token:
            hub_kwargs["token"] = token

        config_path = hf_hub_download(
            self._variant_config.pretrained_model_name,
            "config.yaml",
            **hub_kwargs,
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)
        segmentation_name = config["pipeline"]["params"]["segmentation"]

        self._model = Model.from_pretrained(segmentation_name, **hub_kwargs)
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
        # 10 seconds of mono audio at 16kHz
        waveform = torch.randn(1, 1, 160000, dtype=dtype)
        return [waveform]
