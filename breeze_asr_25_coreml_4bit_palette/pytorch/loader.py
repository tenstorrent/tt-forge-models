# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Breeze-ASR-25 CoreML 4-bit palette model loader implementation.

weiren119/Breeze-ASR-25-coreml-4bit-palette is a 4-bit palette quantized
CoreML conversion of MediaTek-Research/Breeze-ASR-25, which itself is a
zh/en fine-tune of openai/whisper-large-v2. The CoreML artifact has no
PyTorch weights, so this loader uses the upstream transformers-compatible
MediaTek-Research/Breeze-ASR-25 checkpoint for evaluation.
"""

import numpy as np
import torch
from transformers import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from typing import Optional

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
    """Available Breeze-ASR-25 model variants."""

    BREEZE_ASR_25 = "Breeze-ASR-25"


class ModelLoader(ForgeModel):
    """Breeze-ASR-25 CoreML 4-bit palette model loader implementation."""

    _VARIANTS = {
        ModelVariant.BREEZE_ASR_25: ModelConfig(
            pretrained_model_name="MediaTek-Research/Breeze-ASR-25",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BREEZE_ASR_25

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Breeze-ASR-25",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Breeze-ASR-25 model from Hugging Face."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.processor = WhisperProcessor.from_pretrained(pretrained_model_name)

        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for the Breeze-ASR-25 model."""
        if self.model is None or self.processor is None:
            self.load_model()

        model_config = WhisperConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        # Generate synthetic audio (30 seconds at 16kHz to match Whisper's receptive field)
        sample_audio = np.random.randn(16000 * 30).astype(np.float32)
        processor_output = self.processor(
            sample_audio, return_tensors="pt", sampling_rate=16000
        )
        input_features = processor_output.input_features.to(device=device, dtype=dtype)

        decoder_input_ids = torch.full(
            (1, 2),
            model_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        return [input_features, decoder_input_ids]
