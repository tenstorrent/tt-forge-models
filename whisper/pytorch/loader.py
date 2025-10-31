# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper model loader implementation
"""

import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperModel,
    AutoFeatureExtractor,
    WhisperConfig,
)
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ...tools.utils import get_file
from ...base import ForgeModel
from typing import Optional


class ModelVariant(StrEnum):
    """Available Whisper model variants."""

    WHISPER_TINY = "openai/whisper-tiny"
    WHISPER_BASE = "openai/whisper-base"
    WHISPER_SMALL = "openai/whisper-small"
    WHISPER_MEDIUM = "openai/whisper-medium"
    WHISPER_LARGE = "openai/whisper-large"
    WHISPER_LARGE_V3 = "openai/whisper-large-v3"
    WHISPER_LARGE_V3_TURBO = "openai/whisper-large-v3-turbo"


class ModelLoader(ForgeModel):
    """Whisper model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.WHISPER_TINY: ModelConfig(
            pretrained_model_name="openai/whisper-tiny",
        ),
        ModelVariant.WHISPER_BASE: ModelConfig(
            pretrained_model_name="openai/whisper-base",
        ),
        ModelVariant.WHISPER_SMALL: ModelConfig(
            pretrained_model_name="openai/whisper-small",
        ),
        ModelVariant.WHISPER_MEDIUM: ModelConfig(
            pretrained_model_name="openai/whisper-medium",
        ),
        ModelVariant.WHISPER_LARGE: ModelConfig(
            pretrained_model_name="openai/whisper-large",
        ),
        ModelVariant.WHISPER_LARGE_V3: ModelConfig(
            pretrained_model_name="openai/whisper-large-v3",
        ),
        ModelVariant.WHISPER_LARGE_V3_TURBO: ModelConfig(
            pretrained_model_name="openai/whisper-large-v3-turbo",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.WHISPER_TINY

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
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
            model="whisper",
            variant=variant,
            group=(
                ModelGroup.RED
                if variant == ModelVariant.WHISPER_LARGE_V3
                else ModelGroup.GENERALITY
            ),
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.processor = None
        self.feature_extractor = None
        self.model = None

    def load_model(self, dtype_override=None):
        """Load a Whisper model from Hugging Face."""

        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Common model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self._variant == ModelVariant.WHISPER_LARGE_V3:
            self.model = WhisperModel.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                pretrained_model_name
            )
            self.processor = None
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                pretrained_model_name, use_cache=False, **model_kwargs
            )
            self.processor = WhisperProcessor.from_pretrained(
                pretrained_model_name, use_cache=False, **model_kwargs
            )
            self.feature_extractor = None

        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)
        return self.model

    def load_inputs(self):
        """Generate sample inputs for Whisper model."""

        # Ensure model and pre-processing utilities are initialized
        if self.model is None or (
            self.processor is None and self.feature_extractor is None
        ):
            self.load_model()

        model_config = WhisperConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        # Load audio sample
        weights_pth = get_file("test_files/pytorch/whisper/1272-128104-0000.pt")
        sample = torch.load(weights_pth, weights_only=False)
        sample_audio = sample["audio"]["array"]

        # Preprocess audio
        sampling_rate = 16000
        if hasattr(self, "feature_extractor") and self.feature_extractor is not None:
            processed = self.feature_extractor(
                sample_audio, return_tensors="pt", sampling_rate=sampling_rate
            )
        else:
            processed = self.processor(
                sample_audio, return_tensors="pt", sampling_rate=sampling_rate
            )

        input_features = processed.input_features

        # Align inputs and decoder IDs to model's device and dtype
        model_param = next(self.model.parameters())
        decoder_input_ids = torch.tensor([[1, 1]]) * model_config.decoder_start_token_id

        # Common device/dtype alignment
        input_features = input_features.to(
            device=model_param.device, dtype=model_param.dtype
        )
        decoder_input_ids = decoder_input_ids.to(device=model_param.device)

        if self._variant == ModelVariant.WHISPER_LARGE_V3_TURBO:

            encoder_outputs = self.model.model.encoder(input_features)[0].detach()
            return [decoder_input_ids, encoder_outputs]

        return [input_features, decoder_input_ids]
