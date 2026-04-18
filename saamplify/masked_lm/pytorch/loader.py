# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SaAMPLIFY model loader implementation for masked language modeling on protein sequences.

The HuggingFace model's remote code imports xformers (CUDA-only), but only uses it
when running on GPU. We install a pure-PyTorch stub so the import succeeds on CPU.
"""
import importlib
import importlib.machinery
import math
import sys
import types

import torch
import torch.nn as nn


def _ensure_xformers_available():
    """Install a stub xformers package if the real one is unavailable.

    The chandar-lab/SaAMPLIFY_350M model's remote code does
    ``from xformers.ops import SwiGLU, memory_efficient_attention``.
    xformers is CUDA-only, but memory_efficient_attention is only called when
    ``x.is_cuda`` so a pure-PyTorch SwiGLU + a placeholder function suffice.
    """
    try:
        importlib.import_module("xformers.ops")
        return
    except (ImportError, ModuleNotFoundError):
        pass

    class SwiGLU(nn.Module):
        def __init__(self, in_features, hidden_features, out_features, bias=True):
            super().__init__()
            self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
            self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        def forward(self, x):
            return self.w3(nn.functional.silu(self.w1(x)) * self.w2(x))

    def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0):
        scale = 1.0 / math.sqrt(query.shape[-1])
        attn = torch.matmul(query, key.transpose(-2, -1)) * scale
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = torch.softmax(attn, dim=-1)
        if p > 0.0:
            attn = torch.nn.functional.dropout(attn, p=p)
        return torch.matmul(attn, value)

    xformers = types.ModuleType("xformers")
    xformers.__version__ = "0.0.0"
    xformers.__spec__ = importlib.machinery.ModuleSpec("xformers", None)

    ops = types.ModuleType("xformers.ops")
    ops.SwiGLU = SwiGLU
    ops.memory_efficient_attention = memory_efficient_attention
    xformers.ops = ops

    sys.modules["xformers"] = xformers
    sys.modules["xformers.ops"] = ops


_ensure_xformers_available()

from transformers import AutoTokenizer, AutoModel
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


class ModelVariant(StrEnum):
    """Available SaAMPLIFY model variants."""

    SAAMPLIFY_350M = "chandar-lab/SaAMPLIFY_350M"


class ModelLoader(ForgeModel):
    """SaAMPLIFY model loader implementation for masked language modeling on protein sequences."""

    _VARIANTS = {
        ModelVariant.SAAMPLIFY_350M: ModelConfig(
            pretrained_model_name="chandar-lab/SaAMPLIFY_350M",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SAAMPLIFY_350M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SaAMPLIFY",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        _ensure_xformers_available()

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        # Short protein sequence with a mask token for masked LM task
        masked_sequence = "MGSSHHHHHHSSGLVPRGSHM<mask>GSSHHHHHHSSGLVPRGSHM"

        inputs = self.tokenizer(
            masked_sequence,
            return_tensors="pt",
            add_special_tokens=True,
        )

        if "attention_mask" in inputs:
            mask = inputs["attention_mask"].to(torch.float32)
            inputs["attention_mask"] = (1.0 - mask) * torch.finfo(torch.float32).min

        return inputs

    def decode_output(self, outputs, inputs=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_tokens = self.tokenizer.decode(predicted_token_id)

        return predicted_tokens
