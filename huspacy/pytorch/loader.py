# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HuSpaCy Hungarian NLP model loader for sentence embedding generation.

HuSpaCy hu_core_news_md is a medium-sized spaCy pipeline for Hungarian NLP
featuring tok2vec, NER, POS tagging, dependency parsing, and lemmatization.
This loader extracts the static word vectors (200k vectors, 100 dimensions)
and wraps them as a PyTorch embedding model for sentence embedding generation.
"""
import torch
import torch.nn as nn
import spacy
from typing import Optional

from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class HuSpacyEmbeddingModel(nn.Module):
    """Mean-pooling model over pre-computed HuSpaCy token embeddings."""

    def __init__(self, vector_dim: int):
        super().__init__()
        self.proj = nn.Linear(vector_dim, vector_dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        token_embeddings = self.proj(token_embeddings)
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        summed = torch.sum(token_embeddings * mask_expanded, dim=1)
        counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts
        return mean_pooled


class ModelVariant(StrEnum):
    """Available HuSpaCy model variants."""

    HU_CORE_NEWS_MD = "hu_core_news_md"


class ModelLoader(ForgeModel):
    """HuSpaCy model loader for Hungarian sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.HU_CORE_NEWS_MD: LLMModelConfig(
            pretrained_model_name="huspacy/hu_core_news_md",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HU_CORE_NEWS_MD

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._nlp = None
        self.sample_text = "Budapest Magyarország fővárosa és egyben legnagyobb városa."

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="HuSpaCy",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _install_spacy_model(repo_id: str, model_name: str):
        """Download and install a spaCy model from HuggingFace Hub."""
        import zipfile
        import sys

        from huggingface_hub import hf_hub_download

        whl_filename = f"{model_name}-any-py3-none-any.whl"
        whl_path = hf_hub_download(repo_id=repo_id, filename=whl_filename)
        site_packages = [p for p in sys.path if "site-packages" in p][0]
        with zipfile.ZipFile(whl_path, "r") as z:
            z.extractall(site_packages)

    def _load_nlp(self):
        if self._nlp is None:
            repo_id = self._variant_config.pretrained_model_name
            model_name = repo_id.split("/")[-1]
            try:
                self._nlp = spacy.load(model_name)
            except OSError:
                self._install_spacy_model(repo_id, model_name)
                self._nlp = spacy.load(model_name)
        return self._nlp

    def load_model(self, *, dtype_override=None, **kwargs):
        nlp = self._load_nlp()
        vector_dim = nlp.vocab.vectors.shape[1]
        model = HuSpacyEmbeddingModel(vector_dim)
        model.eval()
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        nlp = self._load_nlp()
        doc = nlp.make_doc(self.sample_text)
        max_length = self._variant_config.max_length
        vector_dim = nlp.vocab.vectors.shape[1]

        embeddings = []
        for token in doc:
            embeddings.append(token.vector)

        num_tokens = min(len(embeddings), max_length)
        embeddings = embeddings[:num_tokens]

        if num_tokens < max_length:
            pad_count = max_length - num_tokens
            embeddings.extend([np.zeros(vector_dim, dtype=np.float32)] * pad_count)
            attention_mask = [1] * num_tokens + [0] * pad_count
        else:
            attention_mask = [1] * max_length

        token_embeddings = torch.tensor(
            np.stack(embeddings), dtype=torch.float32
        ).unsqueeze(0)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)

        if dtype_override is not None:
            token_embeddings = token_embeddings.to(dtype=dtype_override)

        return {
            "token_embeddings": token_embeddings,
            "attention_mask": attention_mask,
        }

    def output_postprocess(self, output, inputs=None):
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def decode_output(self, outputs, inputs=None):
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
        if isinstance(fwd_output, torch.Tensor):
            return fwd_output.flatten()
        return fwd_output
