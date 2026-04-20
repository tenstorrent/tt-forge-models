# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Lite-Whisper large-v3-turbo-acc model loader implementation.

Lite-Whisper is a compressed version of OpenAI Whisper produced with LiteASR
(low-rank approximation). The -acc variant targets accuracy retention close to
the base whisper-large-v3-turbo model. The checkpoint ships a custom
``LiteWhisperForConditionalGeneration`` class loaded via ``trust_remote_code``,
and reuses the standard ``openai/whisper-large-v3`` processor.
"""

from typing import Optional

import torch
from transformers import AutoModel, AutoProcessor, WhisperConfig

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
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available Lite-Whisper large-v3-turbo-acc model variants."""

    LARGE_V3_TURBO_ACC = "Large_v3_Turbo_Acc"


class ModelLoader(ForgeModel):
    """Lite-Whisper large-v3-turbo-acc model loader implementation."""

    _VARIANTS = {
        ModelVariant.LARGE_V3_TURBO_ACC: ModelConfig(
            pretrained_model_name="efficient-speech/lite-whisper-large-v3-turbo-acc",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_V3_TURBO_ACC

    # The lite-whisper checkpoint does not bundle a processor; reuse the base
    # openai/whisper-large-v3 processor as documented in the model card.
    PROCESSOR_NAME = "openai/whisper-large-v3"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Lite_Whisper",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Lite-Whisper model from Hugging Face."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(self.PROCESSOR_NAME)

        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for the Lite-Whisper model."""
        if self.model is None or self.processor is None:
            self.load_model()

        model_config = WhisperConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )

        weights_pth = get_file("test_files/pytorch/whisper/1272-128104-0000.pt")
        sample = torch.load(weights_pth, weights_only=False)
        sample_audio = sample["audio"]["array"]

        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

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
