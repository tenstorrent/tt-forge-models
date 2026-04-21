# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GLM-4-Voice-Tokenizer model loader implementation for speech tokenization.
"""

from typing import Optional

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
    """Available GLM-4-Voice-Tokenizer model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """GLM-4-Voice-Tokenizer model loader implementation for speech tokenization."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="zai-org/glm-4-voice-tokenizer",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._feature_extractor = None
        self._decoder_start_token_id = 0

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GLM-4-Voice-Tokenizer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_feature_extractor(self, dtype_override=None):
        from transformers import AutoConfig, WhisperFeatureExtractor

        self._feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        self._decoder_start_token_id = getattr(config, "decoder_start_token_id", 0)

        return self._feature_extractor

    def load_model(self, *, dtype_override=None, **kwargs):
        import torch
        from transformers import AutoModel

        target_dtype = dtype_override if dtype_override is not None else torch.float32
        model_kwargs = {"trust_remote_code": True, "torch_dtype": target_dtype}
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        model.to(target_dtype)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np
        import torch

        if self._feature_extractor is None:
            self._load_feature_extractor(dtype_override=dtype_override)

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        if dtype_override is not None:
            inputs = {
                k: (
                    v.to(dtype_override)
                    if isinstance(v, torch.Tensor) and v.is_floating_point()
                    else v
                )
                for k, v in inputs.items()
            }

        inputs["decoder_input_ids"] = (
            torch.ones((1, 1), dtype=torch.long) * self._decoder_start_token_id
        )

        return inputs
