# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GTE-Multilingual-Base model loader implementation for sentence embedding generation.
"""

import torch
import types
from transformers import AutoModel, AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


def _reinit_non_persistent_buffers(model):
    # transformers 5.x initializes models on meta device; non-persistent buffers are excluded
    # from state_dict so their real-memory tensors contain uninitialized garbage after load.
    model.embeddings.register_buffer(
        "position_ids",
        torch.arange(model.config.max_position_embeddings),
        persistent=False,
    )
    re = model.embeddings.rotary_emb
    inv_freq = 1.0 / (re.base ** (torch.arange(0, re.dim, 2).float() / re.dim))
    re.register_buffer("inv_freq", inv_freq, persistent=False)
    re._set_cos_sin_cache(
        seq_len=re.max_position_embeddings,
        device=inv_freq.device,
        dtype=torch.get_default_dtype(),
    )


def _patched_get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
    # transformers 5.x get_extended_attention_mask uses Python float literals and
    # torch.finfo(dtype).min (both f64 in Python), which XLA traces as f64 constants and
    # promotes the whole attention mask computation to f64. TT hardware does not support f64.
    # Fix: use explicit dtype-typed tensors so the computation stays in the model dtype.
    if dtype is None:
        dtype = self.dtype
    if attention_mask.dim() == 3:
        extended = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )
    extended = extended.to(dtype=dtype)
    one = torch.tensor(1.0, dtype=dtype)
    min_val = torch.tensor(torch.finfo(dtype).min, dtype=dtype)
    return (one - extended) * min_val


class ModelVariant(StrEnum):
    """Available GTE-Multilingual-Base model variants for embedding generation."""

    GTE_MULTILINGUAL_BASE = "gte-multilingual-base"
    TURK_EMBED_4_STS = "newmindai/TurkEmbed4STS"


class ModelLoader(ForgeModel):
    """GTE-Multilingual-Base model loader implementation for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.GTE_MULTILINGUAL_BASE: ModelConfig(
            pretrained_model_name="Alibaba-NLP/gte-multilingual-base",
        ),
        ModelVariant.TURK_EMBED_4_STS: ModelConfig(
            pretrained_model_name="newmindai/TurkEmbed4STS",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GTE_MULTILINGUAL_BASE

    sample_sentences = ["This is an example sentence for generating text embeddings"]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GTE-Multilingual-Base",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        _reinit_non_persistent_buffers(model)
        model.get_extended_attention_mask = types.MethodType(
            _patched_get_extended_attention_mask, model
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
