# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NeoBERT model loader implementation for masked language modeling.
"""

import sys
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SwiGLU(nn.Module):
    """Pure-PyTorch re-implementation of xformers.ops.SwiGLU.

    Weight layout matches the xformers version so that HuggingFace
    ``from_pretrained`` can load the original checkpoint unchanged.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = True,
        *,
        _pack_weights: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w12: Optional[nn.Linear]
        if _pack_weights:
            self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        else:
            self.w12 = None
            self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        self.hidden_features = hidden_features
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.w12 is not None:
            x12 = self.w12(x)
            x1, x2 = x12.chunk(2, dim=-1)
        else:
            x1 = self.w1(x)
            x2 = self.w2(x)
        return self.w3(F.silu(x1) * x2)


def _install_xformers_stub():
    """Register a minimal xformers stub so the NeoBERT remote code can import SwiGLU."""
    if "xformers" not in sys.modules:
        xformers_mod = types.ModuleType("xformers")
        sys.modules["xformers"] = xformers_mod
    else:
        xformers_mod = sys.modules["xformers"]

    ops_mod = types.ModuleType("xformers.ops")
    ops_mod.SwiGLU = _SwiGLU
    sys.modules["xformers.ops"] = ops_mod
    xformers_mod.ops = ops_mod


_install_xformers_stub()

from transformers import AutoModelForMaskedLM, AutoTokenizer
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
    """Available NeoBERT model variants for masked language modeling."""

    NEOBERT_BASE = "chandar-lab/NeoBERT"


class ModelLoader(ForgeModel):
    """NeoBERT model loader implementation for masked language modeling."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.NEOBERT_BASE: LLMModelConfig(
            pretrained_model_name="chandar-lab/NeoBERT",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.NEOBERT_BASE

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.sample_text = "The capital of France is [MASK]."
        self.max_length = 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="NeoBERT",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load NeoBERT model for masked language modeling from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The NeoBERT model instance.
        """

        # Initialize tokenizer (NeoBERT uses bert-base-uncased tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(self.model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for NeoBERT masked language modeling.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            # Ensure tokenizer is initialized
            self.load_model(dtype_override=dtype_override)

        # Data preprocessing
        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for masked language modeling."""
        inputs = self.load_inputs()
        logits = co_out[0]
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_token = self.tokenizer.decode(predicted_token_id)
        print("The predicted token for the [MASK] is:", predicted_token)
