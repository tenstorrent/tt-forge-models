# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERTopic Wikipedia (MaartenGr/BERTopic_Wikipedia) model loader.

BERTopic is a topic modeling pipeline built on top of a sentence embedding
model. The MaartenGr/BERTopic_Wikipedia checkpoint was trained on ~1M
Wikipedia pages and uses sentence-transformers/all-MiniLM-L6-v2 as its
underlying embedding backbone. This loader exposes that embedding transformer
as a torch.nn.Module for compile/inference testing.
"""
import torch
from transformers import AutoTokenizer
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available BERTopic Wikipedia model variants."""

    BERTOPIC_WIKIPEDIA = "BERTopic_Wikipedia"


class ModelLoader(ForgeModel):
    """BERTopic Wikipedia model loader for topic-modeling embedding generation."""

    _VARIANTS = {
        ModelVariant.BERTOPIC_WIKIPEDIA: LLMModelConfig(
            pretrained_model_name="MaartenGr/BERTopic_Wikipedia",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BERTOPIC_WIKIPEDIA

    # Embedding backbone referenced by the BERTopic config.json
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    sample_sentences = [
        "BERTopic is a topic modeling technique that leverages transformer embeddings."
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.topic_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="BERTopic_Wikipedia",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from bertopic import BERTopic

        self.topic_model = BERTopic.load(self._variant_config.pretrained_model_name)

        # BERTopic wraps a sentence_transformers.SentenceTransformer. The first
        # pipeline module is the Transformer wrapper exposing `.auto_model`.
        sentence_transformer = self.topic_model.embedding_model.embedding_model
        transformer_module = sentence_transformer[0]
        self.tokenizer = transformer_module.tokenizer

        model = transformer_module.auto_model

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = getattr(self._variant_config, "max_length", 128)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs

    def output_postprocess(self, output, inputs=None):
        if inputs is None:
            inputs = self.load_inputs()

        attention_mask = inputs["attention_mask"]

        if isinstance(output, (tuple, list)):
            token_embeddings = output[0]
        elif hasattr(output, "last_hidden_state"):
            token_embeddings = output.last_hidden_state
        else:
            token_embeddings = output

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sentence_embeddings

    def decode_output(self, outputs, inputs=None):
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
        tensors = []

        if hasattr(fwd_output, "last_hidden_state"):
            tensors.append(fwd_output.last_hidden_state.flatten())
        if (
            hasattr(fwd_output, "pooler_output")
            and fwd_output.pooler_output is not None
        ):
            tensors.append(fwd_output.pooler_output.flatten())

        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
