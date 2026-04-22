# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Parler-TTS model loader implementation for text-to-speech tasks.
"""
import torch.nn as nn
from transformers import AutoTokenizer
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


class ParlerTTSWrapper(nn.Module):
    """Wrapper around ParlerTTSForConditionalGeneration.

    Exposes the encoder-decoder forward pass that takes tokenized description
    and decoder input IDs, returning logits for audio code prediction.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        return outputs.logits


class ModelVariant(StrEnum):
    """Available Parler-TTS model variants."""

    MINI_V0_1 = "mini_v0.1"
    MINI_V1_1 = "mini-v1.1"


class ModelLoader(ForgeModel):
    """Parler-TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.MINI_V0_1: ModelConfig(
            pretrained_model_name="parler-tts/parler_tts_mini_v0.1",
        ),
        ModelVariant.MINI_V1_1: ModelConfig(
            pretrained_model_name="parler-tts/parler-tts-mini-v1.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINI_V0_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self._num_codebooks = None
        self._pad_token_id = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Parler-TTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import pathlib
        import sys

        # The local parler_tts/ model directory shadows the parler-tts pip package.
        # Temporarily remove the models root from sys.path so the installed package
        # is found instead.
        models_root = str(pathlib.Path(__file__).resolve().parents[2])
        original_path = sys.path.copy()
        sys.path = [p for p in sys.path if p != models_root]
        cached = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k == "parler_tts" or k.startswith("parler_tts.")
        }
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from parler_tts.configuration_parler_tts import ParlerTTSConfig
            from transformers import AutoConfig

            # Register the custom parler_tts config so AutoConfig.from_pretrained
            # can resolve model_type="parler_tts" from the checkpoint config.json.
            try:
                AutoConfig.register("parler_tts", ParlerTTSConfig)
            except ValueError:
                pass  # already registered
        finally:
            sys.path = original_path
            sys.modules.update(cached)

        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        full_model = ParlerTTSForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self._num_codebooks = full_model.decoder.num_codebooks
        self._pad_token_id = full_model.generation_config.pad_token_id
        model = ParlerTTSWrapper(full_model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        import torch

        description = (
            "A female speaker with a slightly low-pitched voice delivers her words"
            " quite expressively, in a very confined sounding environment with clear"
            " audio quality. She speaks very fast."
        )

        description_tokens = self.tokenizer(description, return_tensors="pt")

        # decoder_input_ids must be shape (batch_size * num_codebooks, seq_len)
        # with audio codebook pad tokens — NOT text tokens.
        num_codebooks = self._num_codebooks or 9
        pad_token_id = self._pad_token_id or 1024
        decoder_input_ids = (
            torch.ones(num_codebooks, 1, dtype=torch.long) * pad_token_id
        )

        return (
            description_tokens["input_ids"],
            description_tokens["attention_mask"],
            decoder_input_ids,
        )
