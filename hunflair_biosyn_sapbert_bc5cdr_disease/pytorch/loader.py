# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HunFlair BioSyn SapBERT BC5CDR disease entity mention linker loader.
"""

import importlib
import os
import sys
from typing import Optional

import torch

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


class ModelVariant(StrEnum):
    """Available HunFlair BioSyn SapBERT BC5CDR disease model variants."""

    BIOSYN_SAPBERT_BC5CDR_DISEASE = "biosyn-sapbert-bc5cdr-disease"


class _EmbeddingWrapper(torch.nn.Module):
    """Wraps TransformerDocumentEmbeddings to expose a standard forward()."""

    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

    def forward(self, input_ids, attention_mask):
        out = self.embedding_model(input_ids, attention_mask=attention_mask)
        return out["document_embeddings"]


class ModelLoader(ForgeModel):
    """HunFlair BioSyn SapBERT BC5CDR disease entity mention linker loader."""

    _VARIANTS = {
        ModelVariant.BIOSYN_SAPBERT_BC5CDR_DISEASE: ModelConfig(
            pretrained_model_name="hunflair/biosyn-sapbert-bc5cdr-disease",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BIOSYN_SAPBERT_BC5CDR_DISEASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = (
            "The mutation in the ABCD1 gene causes X-linked adrenoleukodystrophy, "
            "a neurodegenerative disease."
        )
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="HunFlair_BioSyn_SapBERT_BC5CDR_Disease",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _fix_flair_path():
        # The local flair/ model-group dir shadows the PyPI flair package.
        # Temporarily remove any sys.path entry that has flair/ without flair/models/.
        stubs = [
            p
            for p in list(sys.path)
            if p
            and "site-packages" not in p
            and os.path.isdir(os.path.join(p, "flair"))
            and not os.path.isdir(os.path.join(p, "flair", "models"))
        ]
        for p in stubs:
            sys.path.remove(p)
        for key in [k for k in sys.modules if k == "flair" or k.startswith("flair.")]:
            del sys.modules[key]
        importlib.invalidate_caches()
        return stubs

    @staticmethod
    def _restore_flair_path(stubs):
        for p in reversed(stubs):
            sys.path.insert(0, p)

    def load_model(self, *, dtype_override=None, **kwargs):
        stubs = self._fix_flair_path()
        try:
            from flair.models import EntityMentionLinker
        finally:
            self._restore_flair_path(stubs)

        linker = EntityMentionLinker.load(self.model_name)
        dense_emb = linker.candidate_generator.embeddings["dense"]
        self._tokenizer = dense_emb.tokenizer

        model = _EmbeddingWrapper(dense_emb)
        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self._tokenizer is None:
            raise RuntimeError("load_model() must be called before load_inputs()")
        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        if dtype_override is not None:
            inputs = {
                k: v.to(dtype_override) if v.is_floating_point() else v
                for k, v in inputs.items()
            }
        return (inputs["input_ids"], inputs["attention_mask"])

    def decode_output(self, co_out):
        if isinstance(co_out, torch.Tensor):
            print(f"Embedding shape: {co_out.shape}")
            return co_out
        return co_out
