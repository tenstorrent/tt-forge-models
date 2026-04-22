# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AudioSeal model loader implementation for speech watermarking.
"""

from typing import Optional

import torch

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


class ModelVariant(StrEnum):
    """Available AudioSeal model variants."""

    WM_16BITS = "audioseal_wm_16bits"
    DETECTOR_16BITS = "audioseal_detector_16bits"


class ModelLoader(ForgeModel):
    """AudioSeal model loader implementation for speech watermarking."""

    _VARIANTS = {
        ModelVariant.WM_16BITS: ModelConfig(
            pretrained_model_name="audioseal_wm_16bits",
        ),
        ModelVariant.DETECTOR_16BITS: ModelConfig(
            pretrained_model_name="audioseal_detector_16bits",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WM_16BITS

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="AudioSeal",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import sys

        # The local 'audioseal/' model directory shadows the installed audioseal
        # package. Temporarily remove the project root from sys.path and clear any
        # cached namespace entries so the real package is found.
        project_root = str(__import__("pathlib").Path(__file__).resolve().parents[2])
        original_path = sys.path.copy()
        sys.path = [p for p in sys.path if p != project_root]
        cached_audioseal = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k == "audioseal" or k.startswith("audioseal.")
        }
        try:
            from audioseal import AudioSeal
        finally:
            sys.path = original_path
            sys.modules.update(cached_audioseal)

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self._variant == ModelVariant.DETECTOR_16BITS:
            model = AudioSeal.load_detector(pretrained_model_name)
        else:
            model = AudioSeal.load_generator(pretrained_model_name)

        model.eval()
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        # AudioSeal expects 16kHz mono audio as (batch, channels, samples).
        sample_rate = 16000
        duration_seconds = 1
        dtype = dtype_override if dtype_override is not None else torch.float32

        waveform = torch.randn(
            batch_size, 1, sample_rate * duration_seconds, dtype=dtype
        )

        return {"x": waveform, "sample_rate": sample_rate}
