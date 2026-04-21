# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
DiariZen WavLM-Conformer model loader for speaker diarization (PyTorch).
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
    """Available DiariZen PyTorch speaker diarization model variants."""

    WAVLM_LARGE_S80_MD = "BUT-FIT/diarizen-wavlm-large-s80-md"


class ModelLoader(ForgeModel):
    """DiariZen model loader implementation for speaker diarization (PyTorch)."""

    _VARIANTS = {
        ModelVariant.WAVLM_LARGE_S80_MD: ModelConfig(
            pretrained_model_name="BUT-FIT/diarizen-wavlm-large-s80-md",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WAVLM_LARGE_S80_MD

    # Segment size used by the DiariZen pipeline (seconds) and sample rate.
    seg_duration = 16
    sample_rate = 16000

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DiariZen",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from diarizen.pipelines.inference import DiariZenPipeline

        pipeline = DiariZenPipeline.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        model = pipeline._segmentation.model
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        # Generate a synthetic mono waveform matching the pipeline's
        # seg_duration (16s) at 16 kHz. Shape: (batch, channels, samples).
        num_samples = self.sample_rate * self.seg_duration
        waveform = torch.randn(1, 1, num_samples, dtype=torch.float32)

        if dtype_override is not None:
            waveform = waveform.to(dtype_override)

        return waveform
