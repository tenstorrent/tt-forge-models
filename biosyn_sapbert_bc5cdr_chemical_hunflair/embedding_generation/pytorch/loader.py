# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Biosyn Sapbert BC5CDR Chemical (HunFlair) model loader implementation for
biomedical entity embedding generation.

Loads the underlying BERT transformer directly from dmis-lab/biosyn-sapbert-bc5cdr-chemical,
which is the model wrapped by the hunflair flair checkpoint.
"""

from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer

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
    """Available Biosyn Sapbert BC5CDR Chemical (HunFlair) model variants."""

    BIOSYN_SAPBERT_BC5CDR_CHEMICAL = "hunflair/biosyn-sapbert-bc5cdr-chemical"


class ModelLoader(ForgeModel):
    """Loader for hunflair/biosyn-sapbert-bc5cdr-chemical biomedical entity linker."""

    # The underlying transformer that the flair EntityMentionLinker wraps.
    _UNDERLYING_TRANSFORMER = "dmis-lab/biosyn-sapbert-bc5cdr-chemical"

    _VARIANTS = {
        ModelVariant.BIOSYN_SAPBERT_BC5CDR_CHEMICAL: LLMModelConfig(
            pretrained_model_name="hunflair/biosyn-sapbert-bc5cdr-chemical",
            max_length=25,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BIOSYN_SAPBERT_BC5CDR_CHEMICAL

    sample_text = "acetaminophen"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Biosyn Sapbert BC5CDR Chemical (HunFlair)",
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
        if self.tokenizer is None:
            self._load_tokenizer()

        model = AutoModel.from_pretrained(self._UNDERLYING_TRANSFORMER)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        self.model = model
        return model

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
