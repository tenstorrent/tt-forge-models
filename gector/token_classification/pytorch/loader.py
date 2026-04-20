# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GECToR model loader implementation for token classification.

GECToR (Grammatical Error Correction: Tag, Not Rewrite) is a sequence tagging
approach to grammatical error correction. The `GECToR` modeling class is
provided by the `gector` package (https://github.com/gotutiyan/gector) and
wraps a transformer encoder with two linear heads for per-token edit tag and
error detection prediction.
"""

from typing import Optional

from transformers import AutoTokenizer

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
    """Available GECToR model variants for token classification."""

    GOTUTIYAN_GECTOR_DEBERTA_LARGE_5K = "gotutiyan/gector-deberta-large-5k"


class ModelLoader(ForgeModel):
    """GECToR model loader implementation for token classification."""

    _VARIANTS = {
        ModelVariant.GOTUTIYAN_GECTOR_DEBERTA_LARGE_5K: LLMModelConfig(
            pretrained_model_name="gotutiyan/gector-deberta-large-5k",
            max_length=80,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GOTUTIYAN_GECTOR_DEBERTA_LARGE_5K

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.sample_text = "This are a wrong sentences"
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GECToR",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GECToR model.

        Requires the `gector` package:
            pip install gector
        """
        from gector.modeling import GECToR

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = GECToR.from_pretrained(self.model_name, **model_kwargs)
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs
