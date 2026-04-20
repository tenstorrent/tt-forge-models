# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
facebook/MEXMA model loader for cross-lingual sentence embedding generation.

MEXMA is an XLM-RoBERTa-Large based cross-lingual sentence encoder. The pooler
layer was not trained, so the CLS token of the last hidden state is used as the
sentence representation.
"""
import torch
from transformers import AutoTokenizer, XLMRobertaModel
from typing import Optional

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available model variants for facebook/MEXMA."""

    FACEBOOK_MEXMA = "facebook/MEXMA"


class ModelLoader(ForgeModel):
    """facebook/MEXMA model loader."""

    _VARIANTS = {
        ModelVariant.FACEBOOK_MEXMA: LLMModelConfig(
            pretrained_model_name="facebook/MEXMA",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FACEBOOK_MEXMA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="facebook-mexma",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"add_pooling_layer": False}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = XLMRobertaModel.from_pretrained(model_name, **model_kwargs)
        model.eval()

        self.model = model

        return model

    def load_inputs(self, dtype_override=None, sentence=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if sentence is None:
            sentence = ["Sentence1", "Sentence2"]

        max_length = getattr(self._variant_config, "max_length", 128)

        inputs = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def output_postprocess(self, output, inputs=None):
        if isinstance(output, (tuple, list)):
            token_embeddings = output[0]
        elif hasattr(output, "last_hidden_state"):
            token_embeddings = output.last_hidden_state
        else:
            token_embeddings = output

        sentence_embeddings = token_embeddings[:, 0]
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
