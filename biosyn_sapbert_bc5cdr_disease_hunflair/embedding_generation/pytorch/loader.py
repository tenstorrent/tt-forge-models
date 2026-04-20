# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Biosyn Sapbert BC5CDR Disease (HunFlair) model loader implementation for
biomedical entity embedding generation.

The hunflair/biosyn-sapbert-bc5cdr-disease repository ships a pickled flair
EntityMentionLinker. It wraps the BERT transformer from the underlying
dmis-lab/biosyn-sapbert-bc5cdr-disease model. This loader loads the linker via
flair, extracts the underlying transformer, and exposes it as a standard
PyTorch module driven by tokenized tensor inputs.
"""

from typing import Optional

import torch
from transformers import AutoTokenizer

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
    """Available Biosyn Sapbert BC5CDR Disease (HunFlair) model variants."""

    BIOSYN_SAPBERT_BC5CDR_DISEASE = "hunflair/biosyn-sapbert-bc5cdr-disease"


class ModelLoader(ForgeModel):
    """Loader for hunflair/biosyn-sapbert-bc5cdr-disease biomedical entity linker."""

    # The underlying transformer that the flair EntityMentionLinker wraps.
    _UNDERLYING_TRANSFORMER = "dmis-lab/biosyn-sapbert-bc5cdr-disease"

    _VARIANTS = {
        ModelVariant.BIOSYN_SAPBERT_BC5CDR_DISEASE: LLMModelConfig(
            pretrained_model_name="hunflair/biosyn-sapbert-bc5cdr-disease",
            max_length=25,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BIOSYN_SAPBERT_BC5CDR_DISEASE

    sample_text = "adrenoleukodystrophy"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Biosyn Sapbert BC5CDR Disease (HunFlair)",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self._UNDERLYING_TRANSFORMER)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from flair.models import EntityMentionLinker

        if self.tokenizer is None:
            self._load_tokenizer()

        linker = EntityMentionLinker.load(self._variant_config.pretrained_model_name)

        # EntityMentionLinker wraps a SemanticCandidateSearchIndex whose dense
        # embedding is a TransformerDocumentEmbeddings holding the HF model.
        dense_embeddings = linker.candidate_generator.embeddings["dense"]
        transformer = dense_embeddings.model

        if dtype_override is not None:
            transformer = transformer.to(dtype=dtype_override)

        transformer.eval()
        self.model = transformer
        return transformer

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_text,
            padding="max_length",
            truncation=True,
            max_length=self._variant_config.max_length,
            return_tensors="pt",
        )
        return inputs

    def decode_output(self, outputs, inputs=None):
        """Mean-pool the last hidden state to produce an entity embedding."""
        if inputs is None:
            inputs = self.load_inputs()

        if hasattr(outputs, "last_hidden_state"):
            token_embeddings = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            token_embeddings = outputs[0]
        else:
            token_embeddings = outputs

        attention_mask = inputs["attention_mask"]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
