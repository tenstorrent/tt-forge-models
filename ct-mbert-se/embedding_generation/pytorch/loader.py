# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Ct Mbert Se model loader implementation for embedding generation."""
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer

from third_party.tt_forge_models.base import ForgeModel
from third_party.tt_forge_models.config import (
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="crisistransformers/CT-mBERT-SE",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT
    sample_sentences = ["The cat sits on the mat"]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Ct Mbert Se",
            variant=variant or cls.DEFAULT_VARIANT,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._variant_config.pretrained_model_name)
        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {"return_dict": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        tokenizer = self._load_tokenizer()
        inputs = tokenizer(
            self.sample_sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)
        return inputs

    def decode_output(self, outputs, inputs=None):
        if inputs is None:
            inputs = self.load_inputs()
        token_embeddings = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
