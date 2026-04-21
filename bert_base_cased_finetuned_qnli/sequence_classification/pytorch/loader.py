# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERT Base Cased finetuned on QNLI model loader implementation for sequence classification.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
    """Available BERT Base Cased QNLI model variants."""

    BERT_BASE_CASED_FINETUNED_QNLI = "bert-base-cased-finetuned-qnli"


class ModelLoader(ForgeModel):
    """BERT Base Cased finetuned on QNLI model loader for sequence classification."""

    _VARIANTS = {
        ModelVariant.BERT_BASE_CASED_FINETUNED_QNLI: ModelConfig(
            pretrained_model_name="gchhablani/bert-base-cased-finetuned-qnli",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BERT_BASE_CASED_FINETUNED_QNLI

    # Sample question-sentence pair for QNLI testing
    sample_pairs = [
        (
            "What is the capital of France?",
            "Paris is the capital and most populous city of France.",
        ),
    ]

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
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="BERT-Base-Cased-QNLI",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BERT Base Cased QNLI model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The BERT Base Cased QNLI model instance for sequence classification.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the BERT Base Cased QNLI model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        questions = [pair[0] for pair in self.sample_pairs]
        sentences = [pair[1] for pair in self.sample_pairs]

        inputs = self.tokenizer(
            questions,
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float32:
                        inputs[key] = value.to(dtype_override)

        return inputs
