# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tiny WhisperForConditionalGeneration LoRA model loader implementation for speech recognition.
"""

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperConfig,
)
from peft import PeftModel
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
    """Available Tiny WhisperForConditionalGeneration LoRA model variants."""

    TINY_WHISPER_LORA = "Tiny_Whisper_LoRA"


class ModelLoader(ForgeModel):
    """Tiny WhisperForConditionalGeneration LoRA model loader for speech recognition tasks."""

    _VARIANTS = {
        ModelVariant.TINY_WHISPER_LORA: ModelConfig(
            pretrained_model_name="peft-internal-testing/tiny_WhisperForConditionalGeneration-lora",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_WHISPER_LORA

    BASE_MODEL_NAME = "hf-internal-testing/tiny-random-WhisperForConditionalGeneration"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Tiny_WhisperForConditionalGeneration_LoRA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load a Tiny WhisperForConditionalGeneration model with LoRA adapter."""
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        base_model = WhisperForConditionalGeneration.from_pretrained(
            self.BASE_MODEL_NAME, use_cache=False, **model_kwargs
        )

        adapter_name = self._variant_config.pretrained_model_name
        model = PeftModel.from_pretrained(base_model, adapter_name)
        model = model.merge_and_unload()

        for param in model.parameters():
            param.requires_grad = False

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Generate synthetic sample inputs for the model."""
        if self.model is None:
            self.load_model()

        model_config = WhisperConfig.from_pretrained(self.BASE_MODEL_NAME)
        model_param = next(self.model.parameters())
        device = model_param.device
        dtype = dtype_override or model_param.dtype

        # Synthetic mel-spectrogram: (batch, num_mel_bins, 2 * max_source_positions)
        seq_len = 2 * model_config.max_source_positions
        input_features = torch.zeros(
            (1, model_config.num_mel_bins, seq_len), dtype=dtype, device=device
        )

        decoder_input_ids = torch.full(
            (1, 2),
            model_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        return {
            "input_features": input_features,
            "decoder_input_ids": decoder_input_ids,
        }
