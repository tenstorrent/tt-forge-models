# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AMPLIFY model loader implementation for masked language modeling on protein sequences.
"""
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Optional


def _ensure_xformers_stub():
    """Inject a minimal xformers stub so the AMPLIFY hub code can import without CUDA xformers.

    The AMPLIFY remote model code uses xformers.ops.SwiGLU (weight layout: w12 + w3) and
    xformers.ops.memory_efficient_attention (only invoked on CUDA; CPU path uses
    scaled_dot_product_attention). We provide compatible pure-PyTorch replacements.
    """
    if "xformers" in sys.modules:
        return

    class SwiGLU(nn.Module):
        def __init__(
            self, in_features, hidden_features, out_features=None, bias=True, **kwargs
        ):
            super().__init__()
            out_features = out_features or in_features
            self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
            self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        def forward(self, x):
            x1, x2 = self.w12(x).chunk(2, dim=-1)
            return self.w3(F.silu(x1) * x2)

    def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, **kwargs):
        # Transpose from (B, M, H, K) to (B, H, M, K) for sdpa
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=p)
        return out.transpose(1, 2)

    ops_mod = types.ModuleType("xformers.ops")
    ops_mod.SwiGLU = SwiGLU
    ops_mod.memory_efficient_attention = memory_efficient_attention

    xformers_mod = types.ModuleType("xformers")
    xformers_mod.ops = ops_mod

    sys.modules["xformers"] = xformers_mod
    sys.modules["xformers.ops"] = ops_mod


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
    """Available AMPLIFY model variants."""

    AMPLIFY_350M = "chandar-lab/AMPLIFY_350M"


class ModelLoader(ForgeModel):
    """AMPLIFY model loader implementation for masked language modeling on protein sequences."""

    _VARIANTS = {
        ModelVariant.AMPLIFY_350M: ModelConfig(
            pretrained_model_name="chandar-lab/AMPLIFY_350M",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AMPLIFY_350M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AMPLIFY",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        _ensure_xformers_stub()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        _ensure_xformers_stub()
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

        # AMPLIFY expects an additive attention mask (0.0 = attend, -inf = mask)
        inputs["attention_mask"] = torch.where(
            inputs["attention_mask"].bool(),
            torch.tensor(0.0),
            torch.tensor(float("-inf")),
        )

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
