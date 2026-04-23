# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
raxtemur/SONAR_200_text_decoder model loader implementation for text translation.

A HuggingFace port of Meta's SONAR text decoder that converts 1024-dimensional
sentence embeddings back to text across the 202 FLORES-200 languages. The
architecture is an M2M100ForConditionalGeneration initialized from
facebook/nllb-200-distilled-1.3B; at inference the encoder is bypassed and the
SONAR sentence embeddings are injected directly via ``encoder_outputs``.
"""

from typing import Optional

import torch
from transformers import M2M100ForConditionalGeneration, NllbTokenizer

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
    """Available raxtemur/SONAR_200_text_decoder variants."""

    SONAR_200_TEXT_DECODER = "SONAR_200_text_decoder"


class ModelLoader(ForgeModel):
    """raxtemur/SONAR_200_text_decoder loader for text translation tasks."""

    _VARIANTS = {
        ModelVariant.SONAR_200_TEXT_DECODER: LLMModelConfig(
            pretrained_model_name="raxtemur/SONAR_200_text_decoder",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SONAR_200_TEXT_DECODER

    # SONAR sentence embeddings are 1024-dimensional.
    embedding_dim = 1024

    # Target language for decoding; FLORES-200 code for English in Latin script.
    target_lang = "eng_Latn"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="raxtemur SONAR_200_text_decoder",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TRANSLATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = NllbTokenizer.from_pretrained(pretrained_model_name)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = M2M100ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # SONAR decoder consumes sentence embeddings via encoder_outputs rather
        # than input_ids. Shape is [batch, 1, embedding_dim] since the embedding
        # represents a single sentence.
        embeddings = torch.randn(batch_size, 1, self.embedding_dim)
        if dtype_override is not None:
            embeddings = embeddings.to(dtype_override)

        # Tuple rather than BaseModelOutput: transformers 5.x M2M100Model.forward does
        # `decoder_outputs + encoder_outputs` when return_dict=False, which fails if
        # encoder_outputs is a BaseModelOutput. A tuple works for both return_dict modes.
        encoder_outputs = (embeddings,)

        # Seq2seq decoding starts from the target language BOS token.
        target_lang_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)
        decoder_input_ids = torch.tensor([[target_lang_id]]).repeat_interleave(
            batch_size, dim=0
        )

        return {
            "encoder_outputs": encoder_outputs,
            "decoder_input_ids": decoder_input_ids,
        }
