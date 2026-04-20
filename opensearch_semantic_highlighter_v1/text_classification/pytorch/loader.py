# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenSearch Semantic Highlighter v1 model loader for sentence-level text classification.
"""

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
from .src.model_utils import (
    BertTaggerForSentenceExtractionWithBackoff,
    prepare_highlighter_inputs,
)


class ModelVariant(StrEnum):
    """Available OpenSearch Semantic Highlighter v1 model variants."""

    SEMANTIC_HIGHLIGHTER_V1 = "Semantic_Highlighter_V1"


class ModelLoader(ForgeModel):
    """OpenSearch Semantic Highlighter v1 model loader implementation."""

    _VARIANTS = {
        ModelVariant.SEMANTIC_HIGHLIGHTER_V1: LLMModelConfig(
            pretrained_model_name="opensearch-project/opensearch-semantic-highlighter-v1",
            max_length=510,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEMANTIC_HIGHLIGHTER_V1

    sample_query = "When does OpenSearch use text reanalysis for highlighting?"
    sample_document_sentences = [
        "To highlight the search terms, the highlighter needs the start and end character offsets of each term.",
        "The offsets mark the term's position in the original text.",
        "The highlighter can obtain the offsets from the following sources: postings, term vectors, or text reanalysis.",
        "Postings: when documents are indexed, OpenSearch creates an inverted search index used to search for documents.",
        "If you set the index_options parameter to offsets when mapping a text field, OpenSearch adds each term's start and end character offsets to the inverted index.",
        "Text reanalysis: in the absence of both postings and term vectors, the highlighter reanalyzes text in order to highlight it.",
        "Reanalyzing the text works well in most use cases, but is more memory and time intensive for large fields.",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""

        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""

        return ModelInfo(
            model="OpenSearch Semantic Highlighter",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant."""

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the OpenSearch Semantic Highlighter v1 model instance."""

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BertTaggerForSentenceExtractionWithBackoff.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model."""

        if self.tokenizer is None:
            self._load_tokenizer()

        return prepare_highlighter_inputs(
            self.tokenizer,
            self.sample_query,
            self.sample_document_sentences,
            max_seq_length=self._variant_config.max_length,
        )
