# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Diffusion Cosine BabyLM model loader implementation for masked language modeling.
"""
import json

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Diffusion Cosine BabyLM model variants."""

    DIFFUSION_COSINE_BABYLM = "despoinakk/diffusion_cosine_babylm"


class ModelLoader(ForgeModel):
    """Diffusion Cosine BabyLM model loader for masked language modeling."""

    _VARIANTS = {
        ModelVariant.DIFFUSION_COSINE_BABYLM: LLMModelConfig(
            pretrained_model_name="despoinakk/diffusion_cosine_babylm",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DIFFUSION_COSINE_BABYLM

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Diffusion Cosine BabyLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        # The remote model's custom config to_json_string() cannot serialize
        # torch.dtype, so we temporarily patch json.JSONEncoder.default.
        _original_default = json.JSONEncoder.default

        def _dtype_aware_default(self, obj):
            if isinstance(obj, torch.dtype):
                return str(obj)
            return _original_default(self, obj)

        json.JSONEncoder.default = _dtype_aware_default
        try:
            model = AutoModelForMaskedLM.from_pretrained(
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
                **kwargs,
            )
        finally:
            json.JSONEncoder.default = _original_default

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        sample_text = "The capital of France is [MASK]."

        max_length = self._variant_config.max_length
        inputs = self.tokenizer(
            sample_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_tokens = self.tokenizer.decode(predicted_token_id)

        return predicted_tokens
