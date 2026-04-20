# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hypencoder dual-encoder model loader for dense retrieval feature extraction.
"""
from typing import Optional

import torch
from transformers import AutoTokenizer

from third_party.tt_forge_models.base import ForgeModel
from third_party.tt_forge_models.config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

from .src.model import (
    HypencoderDualEncoder,
    HypencoderDualEncoderConfig,
    HypencoderScoringWrapper,
)


class ModelVariant(StrEnum):
    """Available Hypencoder model variants."""

    HYPENCODER_6_LAYER = "jfkback/hypencoder.6_layer"


class ModelLoader(ForgeModel):
    """Hypencoder dual-encoder model loader for dense retrieval feature extraction."""

    _VARIANTS = {
        ModelVariant.HYPENCODER_6_LAYER: LLMModelConfig(
            pretrained_model_name="jfkback/hypencoder.6_layer",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HYPENCODER_6_LAYER

    sample_query = "how many states are there in india"
    sample_passage = "India has 28 states and 8 union territories."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Hypencoder",
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

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        config = HypencoderDualEncoderConfig.from_pretrained(model_name)
        model = HypencoderDualEncoder.from_pretrained(
            model_name, config=config, **model_kwargs
        )
        model.eval()

        wrapped = HypencoderScoringWrapper(model)
        wrapped.eval()

        self.model = wrapped
        return wrapped

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = getattr(self._variant_config, "max_length", 128)

        query_inputs = self.tokenizer(
            self.sample_query,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        passage_inputs = self.tokenizer(
            self.sample_passage,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "query_input_ids": query_inputs["input_ids"],
            "query_attention_mask": query_inputs["attention_mask"],
            "passage_input_ids": passage_inputs["input_ids"],
            "passage_attention_mask": passage_inputs["attention_mask"],
        }

    def decode_output(self, outputs, inputs=None):
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs

    def unpack_forward_output(self, fwd_output):
        if isinstance(fwd_output, torch.Tensor):
            return fwd_output.flatten()
        return fwd_output
