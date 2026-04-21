# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ruRoberta model loader implementation for sequence classification (paraphrase detection).
"""
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available ruRoberta sequence classification model variants."""

    LARGE_PARAPHRASE_V1 = "Large_Paraphrase_v1"


class ModelLoader(ForgeModel):
    """ruRoberta model loader implementation for sequence classification (paraphrase detection)."""

    _VARIANTS = {
        ModelVariant.LARGE_PARAPHRASE_V1: ModelConfig(
            pretrained_model_name="s-nlp/ruRoberta-large-paraphrase-v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_PARAPHRASE_V1

    # Sample Russian sentence pair for paraphrase detection
    sample_text_1 = "Я тебя люблю"
    sample_text_2 = "Ты мне нравишься"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="ruRoberta",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ruRoberta sequence classification model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The ruRoberta model instance for paraphrase classification.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for paraphrase classification.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
            batch_size: Optional batch size to override the default of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) for the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            [self.sample_text_1] * batch_size,
            [self.sample_text_2] * batch_size,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode model output into a paraphrase probability.

        Args:
            co_out: Model output from a forward pass.

        Returns:
            str: Human-readable paraphrase probability.
        """
        logits = co_out[0] if isinstance(co_out, (list, tuple)) else co_out.logits
        proba = torch.softmax(logits, dim=-1)
        print(f"Paraphrase probability: {proba[0][1].item():.4f}")
