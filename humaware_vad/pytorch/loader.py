# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HumAware-VAD model loader implementation for voice activity detection.

Loads the CuriousMonkey7/HumAware-VAD TorchScript model, a Silero-VAD
fine-tune that reduces false positives from humming and other vocal
sounds.
"""

from typing import Optional

import torch
from huggingface_hub import hf_hub_download

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
    """Available HumAware-VAD model variants."""

    HUMAWARE_VAD = "HumAware-VAD"


class ModelLoader(ForgeModel):
    """HumAware-VAD model loader implementation for voice activity detection."""

    _VARIANTS = {
        ModelVariant.HUMAWARE_VAD: ModelConfig(
            pretrained_model_name="CuriousMonkey7/HumAware-VAD",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUMAWARE_VAD

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="HumAware-VAD",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the HumAware-VAD TorchScript model from HuggingFace Hub."""
        jit_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="HumAwareVAD.jit",
        )
        model = torch.jit.load(jit_path)
        model.eval()

        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the HumAware-VAD model.

        HumAware-VAD is a Silero-VAD fine-tune, which expects a 512-sample
        mono audio chunk at 16kHz (32ms window) along with the sampling rate.
        """
        chunk_size = 512
        sampling_rate = 16000

        audio_chunk = torch.randn(1, chunk_size)

        if dtype_override is not None:
            audio_chunk = audio_chunk.to(dtype_override)

        return [audio_chunk, torch.tensor(sampling_rate)]
