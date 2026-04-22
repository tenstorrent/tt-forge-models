# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Muaalem Model v3.2 loader implementation for Quranic speech recognition (ASR) using PyTorch.
"""

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
    """Available Muaalem Model v3.2 speech recognition model variants."""

    OBADX_MUAALEM_MODEL_V3_2 = "obadx/muaalem-model-v3_2"


class ModelLoader(ForgeModel):
    """Muaalem Model v3.2 loader implementation for speech recognition (PyTorch)."""

    _VARIANTS = {
        ModelVariant.OBADX_MUAALEM_MODEL_V3_2: ModelConfig(
            pretrained_model_name="obadx/muaalem-model-v3_2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OBADX_MUAALEM_MODEL_V3_2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Muaalem-Model-v3_2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoConfig, Wav2Vec2BertConfig, Wav2Vec2BertForCTC

        # Register the custom model type so AutoConfig can resolve it.
        # This checkpoint uses "multi_level_ctc" which is not in standard transformers.
        try:
            AutoConfig.register("multi_level_ctc", Wav2Vec2BertConfig)
        except ValueError:
            pass  # already registered

        config = Wav2Vec2BertConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        # The custom multi-level config stores per-level vocab sizes; set the primary one.
        if config.vocab_size is None:
            level_vocab = getattr(config, "level_to_vocab_size", {})
            config.vocab_size = level_vocab.get("phonemes", 43)

        model_kwargs = {"config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Wav2Vec2BertForCTC.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Generate a synthetic 1-second audio waveform at 16kHz
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
                k: v.to(dtype_override) if v.is_floating_point() else v
                for k, v in inputs.items()
            }

        return inputs
