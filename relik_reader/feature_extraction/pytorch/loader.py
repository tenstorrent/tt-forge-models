# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ReLiK Reader model loader implementation for relation extraction.

ReLiK (Retrieve, Read and Link) is an information-extraction framework whose
reader component combines a transformer encoder (DeBERTa v3 Small here) with
custom heads for entity span detection and relation classification. The
architecture is defined in the model repository itself, so loading requires
``trust_remote_code=True``.
"""

from typing import Optional

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
    """Available ReLiK Reader variants."""

    DEBERTA_V3_SMALL_RE_WIKIPEDIA = "relik-reader-deberta-v3-small-re-wikipedia"


class ModelLoader(ForgeModel):
    """ReLiK Reader model loader for relation extraction."""

    _VARIANTS = {
        ModelVariant.DEBERTA_V3_SMALL_RE_WIKIPEDIA: LLMModelConfig(
            pretrained_model_name="relik-ie/relik-reader-deberta-v3-small-re-wikipedia",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEBERTA_V3_SMALL_RE_WIKIPEDIA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.sample_text = "Michael Jordan was one of the best players in the NBA."
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ReLiK-Reader",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )

        return inputs
