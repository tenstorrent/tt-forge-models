# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PubMedBERT SPLADE model loader implementation for masked language modeling.
"""

from typing import Optional

from transformers import AutoModelForMaskedLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available PubMedBERT SPLADE model variants."""

    PUBMEDBERT_BASE_SPLADE = "NeuML/pubmedbert-base-splade"


class ModelLoader(ForgeModel):
    """PubMedBERT SPLADE model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.PUBMEDBERT_BASE_SPLADE: LLMModelConfig(
            pretrained_model_name="NeuML/pubmedbert-base-splade",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PUBMEDBERT_BASE_SPLADE

    sample_text = (
        "Chronic kidney disease is associated with increased cardiovascular risk."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""

        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""

        return ModelInfo(
            model="PubMedBERT SPLADE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant."""

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the PubMedBERT SPLADE model instance."""

        if self._tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(self._model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model."""

        if self._tokenizer is None:
            self._load_tokenizer()

        inputs = self._tokenizer(
            self.sample_text,
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs
