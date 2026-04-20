# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CHATWhisper-en model loader implementation for speech recognition (ASR).

talkbank/CHATWhisper-en is a fine-tune of openai/whisper-large-v2 optimized for
Language Sample Analysis of English conversational speech.
"""

from typing import Optional

import torch

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
    """Available CHATWhisper-en model variants."""

    EN = "en"


class ModelLoader(ForgeModel):
    """CHATWhisper-en model loader implementation for English conversational ASR."""

    _VARIANTS = {
        ModelVariant.EN: ModelConfig(
            pretrained_model_name="talkbank/CHATWhisper-en",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EN

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """

        super().__init__(variant)
        self.processor = None
        self.model = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="CHATWhisper_en",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the CHATWhisper-en model instance.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """

        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = WhisperForConditionalGeneration.from_pretrained(
            self._model_name, use_cache=False, **model_kwargs
        )
        self.processor = WhisperProcessor.from_pretrained(self._model_name)

        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the CHATWhisper-en model.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        from transformers import WhisperConfig

        if self.model is None or self.processor is None:
            self.load_model()

        whisper_config = WhisperConfig.from_pretrained(self._model_name)

        # Load audio sample
        weights_pth = get_file("test_files/pytorch/whisper/1272-128104-0000.pt")
        sample = torch.load(weights_pth, weights_only=False)
        sample_audio = sample["audio"]["array"]
        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        # Preprocess audio
        sampling_rate = 16000
        inputs = self.processor(
            sample_audio, return_tensors="pt", sampling_rate=sampling_rate
        )
        input_features = inputs.input_features.to(device=device, dtype=dtype)

        # Build decoder input IDs for English transcription
        decoder_prompt_ids = self.processor.get_decoder_prompt_ids(
            task="transcribe", language="en", no_timestamps=True
        )
        init_tokens = [whisper_config.decoder_start_token_id]
        if decoder_prompt_ids:
            init_tokens += [tok for _, tok in decoder_prompt_ids]

        decoder_input_ids = torch.tensor([init_tokens], dtype=torch.long, device=device)
        return [input_features, decoder_input_ids]
