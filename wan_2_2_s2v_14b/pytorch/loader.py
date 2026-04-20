#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan 2.2 Sound-to-Video 14B model loader implementation.

Loads the sharded-safetensors Wan 2.2 Sound-to-Video 14B denoising
transformer from Wan-AI/Wan2.2-S2V-14B. This model generates video
from an audio waveform combined with a reference image and optional
text prompt.

The upstream config.json uses `_class_name: WanModel_S2V`, which is
not yet integrated into the upstream diffusers library, so this
loader downloads the shard index file to expose the model artifacts
without instantiating a PyTorch module.

Available variants:
- WAN22_S2V_14B: Wan 2.2 Sound-to-Video 14B (base safetensors release)
"""

from typing import Any, Optional

import torch
from huggingface_hub import hf_hub_download  # type: ignore[import]

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

REPO_ID = "Wan-AI/Wan2.2-S2V-14B"


class ModelVariant(StrEnum):
    """Available Wan 2.2 S2V 14B model variants."""

    WAN22_S2V_14B = "2.2_S2V_14B"


class ModelLoader(ForgeModel):
    """Wan 2.2 Sound-to-Video 14B model loader.

    Downloads the safetensors shard index for the denoising transformer
    from HuggingFace. The upstream model class (WanModel_S2V) is not yet
    available in diffusers, so the loader exposes the model artifact path
    rather than a PyTorch module.
    """

    _VARIANTS = {
        ModelVariant.WAN22_S2V_14B: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_S2V_14B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model_path = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_2_S2V_14B",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Download and return the path to the sharded safetensors index.

        The WanModel_S2V class referenced by config.json is not yet part of
        upstream diffusers, so the loader returns the local path to the
        safetensors shard index. The index, together with the per-shard
        files resolved via `hf_hub_download`, can be consumed by a custom
        inference pipeline.

        Returns:
            str: Local path to the downloaded safetensors index file.
        """
        self._model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="diffusion_pytorch_model.safetensors.index.json",
        )
        return self._model_path

    def load_inputs(self, **kwargs) -> Any:
        """Prepare dummy inputs for the sound-to-video model.

        Returns a dict with a dummy audio waveform tensor (5 seconds at
        16 kHz) matching the wav2vec2 audio encoder sampling rate.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        sample_rate = 16000
        duration_sec = 5
        return {
            "audio": torch.randn(1, sample_rate * duration_sec, dtype=dtype),
        }
