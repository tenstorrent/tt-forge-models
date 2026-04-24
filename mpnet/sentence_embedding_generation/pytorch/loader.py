# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MPNet model loader for sentence embedding generation (all-mpnet-base-v2).
"""
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
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


class ModelVariant(StrEnum):
    ALL_MPNET_BASE_V2 = "sentence-transformers/all-mpnet-base-v2"


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.ALL_MPNET_BASE_V2: LLMModelConfig(
            pretrained_model_name="sentence-transformers/all-mpnet-base-v2",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ALL_MPNET_BASE_V2

    def __init__(self, variant=None, num_layers=None):
        super().__init__(variant)
        self.model = None
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant=None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MPNet",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, sentence=None):
        if self.tokenizer is None:
            self._load_tokenizer()
        if sentence is None:
            sentence = "The quick brown fox jumps over the lazy dog."
        max_length = getattr(self._variant_config, "max_length", 128)
        return self.tokenizer(
            sentence, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )

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
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def decode_output(self, outputs, inputs=None):
        return self.output_postprocess(outputs, inputs=inputs)
