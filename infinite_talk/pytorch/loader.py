#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InfiniteTalk loader implementation.

InfiniteTalk (MeiGen-AI/InfiniteTalk) is an audio-driven video dubbing
framework that generates lip-synced videos with natural head, body, and
facial expression alignment. The full pipeline is built on Wan2.1-I2V-14B
with a Wav2Vec2 audio encoder for audio conditioning.

Variants:
- SINGLE: Wav2Vec2 audio encoder component (TencentGameMate/chinese-wav2vec2-base)
  used for audio conditioning in the InfiniteTalk pipeline.
- LIGHTWEIGHT_SINGLE_Q4_K_M / LIGHTWEIGHT_MULTI_Q4_K_M: GGUF-quantized
  Wan2.1-InfiniteTalk transformer modules from lightweight/InfiniteTalk,
  loaded as a WanTransformer3DModel via diffusers GGUFQuantizationConfig.
"""

from typing import Any, Optional

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

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

WAV2VEC2_REPO_ID = "TencentGameMate/chinese-wav2vec2-base"
LIGHTWEIGHT_GGUF_REPO_ID = "lightweight/InfiniteTalk"

# Small spatial dimensions for compile-only testing of the transformer.
TRANSFORMER_NUM_FRAMES = 2
TRANSFORMER_HEIGHT = 4
TRANSFORMER_WIDTH = 4
TRANSFORMER_TEXT_SEQ_LEN = 8


class ModelVariant(StrEnum):
    """Available InfiniteTalk model variants."""

    SINGLE = "single"
    LIGHTWEIGHT_SINGLE_Q4_K_M = "lightweight_Single_Q4_K_M"
    LIGHTWEIGHT_MULTI_Q4_K_M = "lightweight_Multi_Q4_K_M"


_LIGHTWEIGHT_GGUF_FILES = {
    ModelVariant.LIGHTWEIGHT_SINGLE_Q4_K_M: "InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q4_K_M.gguf",
    ModelVariant.LIGHTWEIGHT_MULTI_Q4_K_M: "InfiniteTalk/Wan2_1-InfiniteTalk_Multi_Q4_K_M.gguf",
}


class ModelLoader(ForgeModel):
    """InfiniteTalk loader for audio-driven video dubbing."""

    _VARIANTS = {
        ModelVariant.SINGLE: ModelConfig(
            pretrained_model_name=WAV2VEC2_REPO_ID,
        ),
        ModelVariant.LIGHTWEIGHT_SINGLE_Q4_K_M: ModelConfig(
            pretrained_model_name=LIGHTWEIGHT_GGUF_REPO_ID,
        ),
        ModelVariant.LIGHTWEIGHT_MULTI_Q4_K_M: ModelConfig(
            pretrained_model_name=LIGHTWEIGHT_GGUF_REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SINGLE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize InfiniteTalk loader."""
        super().__init__(variant)
        self._processor = None
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="InfiniteTalk",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _is_lightweight_gguf(self) -> bool:
        return self._variant in _LIGHTWEIGHT_GGUF_FILES

    def _load_processor(self):
        """Load the Wav2Vec2 feature extractor for audio preprocessing."""
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self._processor

    def _load_wav2vec2(self, *, dtype_override=None, **kwargs):
        """Load the Wav2Vec2 audio encoder used by InfiniteTalk."""
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Wav2Vec2Model.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        if self._processor is None:
            self._load_processor()

        return model

    def _load_lightweight_transformer(
        self, *, dtype_override: Optional[torch.dtype] = None
    ):
        """Load the GGUF-quantized Wan 2.1 InfiniteTalk transformer module."""
        import diffusers.utils.import_utils as _diffusers_import_utils

        if not _diffusers_import_utils._gguf_available:
            import importlib.util

            if importlib.util.find_spec("gguf") is not None:
                _diffusers_import_utils._gguf_available = True

        from diffusers import GGUFQuantizationConfig, WanTransformer3DModel

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _LIGHTWEIGHT_GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        self._transformer = WanTransformer3DModel.from_single_file(
            f"https://huggingface.co/{LIGHTWEIGHT_GGUF_REPO_ID}/resolve/main/{gguf_file}",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the requested InfiniteTalk component."""
        if self._is_lightweight_gguf():
            return self._load_lightweight_transformer(dtype_override=dtype_override)
        return self._load_wav2vec2(dtype_override=dtype_override, **kwargs)

    def _load_wav2vec2_inputs(self, dtype_override=None):
        """Prepare synthetic audio inputs for the Wav2Vec2 encoder."""
        if self._processor is None:
            self._load_processor()

        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        if dtype_override is not None:
            inputs = {
                k: v.to(dtype_override) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

        return dict(inputs)

    def _load_lightweight_transformer_inputs(
        self, dtype_override: Optional[torch.dtype] = None
    ) -> dict:
        """Prepare tensor inputs for the WanTransformer3DModel forward pass."""
        if self._transformer is None:
            self._load_lightweight_transformer(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        config = self._transformer.config

        return {
            "hidden_states": torch.randn(
                1,
                config.in_channels,
                TRANSFORMER_NUM_FRAMES,
                TRANSFORMER_HEIGHT,
                TRANSFORMER_WIDTH,
                dtype=dtype,
            ),
            "encoder_hidden_states": torch.randn(
                1,
                TRANSFORMER_TEXT_SEQ_LEN,
                config.text_dim,
                dtype=dtype,
            ),
            "timestep": torch.tensor([500], dtype=torch.long),
            "return_dict": False,
        }

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        """Load and return sample inputs for the selected variant."""
        if self._is_lightweight_gguf():
            return self._load_lightweight_transformer_inputs(
                dtype_override=dtype_override
            )
        return self._load_wav2vec2_inputs(dtype_override=dtype_override)
