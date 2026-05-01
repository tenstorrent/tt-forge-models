# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
mRNABERT model loader implementation for embedding generation on mRNA sequences.
"""
import os
import sys
from typing import Optional
from unittest.mock import patch

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module, get_imports

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


def _fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    # flash_attn_triton.py unconditionally imports triton, but bert_layers.py
    # wraps its import in try/except so triton is optional. Transformers'
    # check_imports recurses into relative imports without respecting try/except,
    # so we skip the triton check here. The actual import will fail at runtime
    # and bert_layers.py's except block sets flash_attn_qkvpacked_func=None.
    imports = get_imports(filename)
    if "flash_attn_triton" in str(filename) and "triton" in imports:
        imports.remove("triton")
    return imports


def _patch_bert_padding_for_xla() -> None:
    """Replace bert_padding functions with XLA-compatible plain tensor ops.

    The original implementations use torch.autograd.Function subclasses
    (IndexFirstAxis, IndexPutFirstAxis) and einops.rearrange. When XLA/Dynamo
    traces through them it generates prims::view_of with alias annotations that
    the XLA functionalizer rejects. Plain index/scatter equivalents avoid all
    view-aliased ops in the trace graph.

    bert_layers.py imports these names directly into its own namespace via
    "from .bert_padding import ...", so we must patch both the bert_padding
    module and the bert_layers module's local references.
    """
    import torch.nn.functional as _F

    def _index_first_axis(input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        # input: [N, ...], indices: [nnz] → output: [nnz, ...]
        # Advanced indexing always copies; no view alias created.
        return input[indices]

    def _index_put_first_axis(
        values: torch.Tensor, indices: torch.Tensor, first_axis_dim: int
    ) -> torch.Tensor:
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        output[indices] = values
        return output

    def _unpad_input(
        hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ):
        # Avoid rearrange()+index_first_axis() which produce prims::view_of.
        # Use direct 2D advanced indexing instead.
        seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen = int(seqlens.max().item())
        cu_seqlens = _F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        b, s = hidden_states.shape[0], hidden_states.shape[1]
        hidden_states = hidden_states[indices // s, indices % s]
        return hidden_states, indices, cu_seqlens, max_seqlen

    def _unpad_input_only(
        hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        b, s = hidden_states.shape[0], hidden_states.shape[1]
        return hidden_states[indices // s, indices % s]

    def _pad_input(
        hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int
    ) -> torch.Tensor:
        # Scatter back into a zero-padded [batch*seqlen, ...] buffer, then
        # reshape to [batch, seqlen, ...].  output is a fresh tensor (no alias
        # to hidden_states), so the reshape produces aten::view, not prims::view_of.
        output = torch.zeros(
            batch * seqlen, *hidden_states.shape[1:],
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        output[indices] = hidden_states
        return output.reshape(batch, seqlen, *hidden_states.shape[1:])

    replacements = {
        "index_first_axis": _index_first_axis,
        "index_put_first_axis": _index_put_first_axis,
        "unpad_input": _unpad_input,
        "unpad_input_only": _unpad_input_only,
        "pad_input": _pad_input,
    }

    bert_layers_mod = None
    for name, mod in list(sys.modules.items()):
        if "YYLY66" not in name:
            continue
        if "bert_padding" in name or "bert_layers" in name:
            for attr, fn in replacements.items():
                if hasattr(mod, attr):
                    setattr(mod, attr, fn)
        if "bert_layers" in name:
            bert_layers_mod = mod

    if bert_layers_mod is None:
        return

    # Patch BertUnpadSelfAttention.forward to replace torch.squeeze(attn_mask) == 1.
    # torch.squeeze on a non-squeezable tensor (no size-1 dims) decomposes to
    # prims::view_of in Dynamo via the aten.squeeze.default decomposition. XLA's
    # functionalizer rejects prims::view_of with alias annotations on non-ATen ops.
    # Using attn_mask.ne(0) produces the same boolean mask without any alias.
    _blm = bert_layers_mod
    import math as _math
    import torch.nn as _nn

    def _bert_unpad_self_attention_forward(
        self, hidden_states, cu_seqlens, max_seqlen_in_batch, indices, attn_mask, bias
    ):
        qkv = self.Wqkv(hidden_states)
        qkv = _blm.pad_input(qkv, indices, cu_seqlens.shape[0] - 1, max_seqlen_in_batch)
        qkv = _blm.rearrange(qkv, "b s (t h d) -> b s t h d", t=3, h=self.num_attention_heads)
        flash_fn = _blm.flash_attn_qkvpacked_func
        if self.p_dropout or flash_fn is None:
            q = qkv[:, :, 0, :, :].permute(0, 2, 1, 3)
            k = qkv[:, :, 1, :, :].permute(0, 2, 3, 1)
            v = qkv[:, :, 2, :, :].permute(0, 2, 1, 3)
            attention_scores = torch.matmul(q, k) / _math.sqrt(self.attention_head_size)
            attention_scores = attention_scores + bias
            attention_probs = _nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            attention = torch.matmul(attention_probs, v).permute(0, 2, 1, 3)
        else:
            convert_dtype = qkv.dtype not in [torch.float16, torch.bfloat16]
            if convert_dtype:
                orig_dtype = qkv.dtype
                bias_dtype = bias.dtype
                qkv = qkv.to(torch.float16)
                bias = bias.to(torch.float16)
                attention = flash_fn(qkv, bias)
                attention = attention.to(orig_dtype)
                bias = bias.to(bias_dtype)
            else:
                attention = flash_fn(qkv, bias)
        attention = _blm.unpad_input_only(attention, attn_mask.ne(0))
        return _blm.rearrange(attention, "nnz h d -> nnz (h d)")

    BertUnpadSelfAttention = getattr(bert_layers_mod, "BertUnpadSelfAttention", None)
    if BertUnpadSelfAttention is not None:
        BertUnpadSelfAttention.forward = _bert_unpad_self_attention_forward


def _load_mrnabert_model(model_name: str, **model_kwargs) -> torch.nn.Module:
    """Load mRNABERT bypassing transformers 5.x meta-device initialization.

    BertEncoder computes ALiBi tensors during __init__ (same pattern as
    jina-embeddings-v2 / MosaicBERT). Transformers 5.x initializes models in
    a meta-device context, so torch.arange(..., device=None) becomes a meta
    tensor while torch.Tensor([...]) from a Python list stays on CPU, causing
    a device mismatch in the ALiBi computation. We get the class directly and
    instantiate on CPU, then load weights separately.
    """
    dtype = model_kwargs.pop("torch_dtype", None)

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    with patch("transformers.dynamic_module_utils.get_imports", _fixed_get_imports):
        cls = get_class_from_dynamic_module(
            config.auto_map["AutoModel"],
            model_name,
            trust_remote_code=True,
        )

    _patch_bert_padding_for_xla()

    model = cls(config)

    weights_path = hf_hub_download(model_name, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    # Checkpoint is BertForMaskedLM (bert.* + cls.*); we load BertModel so strip
    # the "bert." prefix and drop the MLM head keys (cls.*).
    state_dict = {k[len("bert."):]: v for k, v in state_dict.items() if k.startswith("bert.")}
    # strict=False: pooler weights are absent from this checkpoint (model was
    # trained without pooler) — all encoder/embedding weights are present.
    model.load_state_dict(state_dict, strict=False)

    if dtype is not None:
        model = model.to(dtype)
        # model.to() does not cast plain attributes, only parameters/buffers.
        # BertEncoder.alibi is a plain attr (not register_buffer), so cast it
        # manually to prevent a float32/bfloat16 mismatch in the attention
        # bias addition (slopes are created via torch.Tensor([...]) → float32).
        if hasattr(model, "encoder") and hasattr(model.encoder, "alibi"):
            model.encoder.alibi = model.encoder.alibi.to(dtype)

    return model


class ModelVariant(StrEnum):
    """Available mRNABERT model variants for embedding generation."""

    MRNABERT = "YYLY66/mRNABERT"


class ModelLoader(ForgeModel):
    """mRNABERT model loader for embedding generation on mRNA sequences."""

    _VARIANTS = {
        ModelVariant.MRNABERT: ModelConfig(
            pretrained_model_name="YYLY66/mRNABERT",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MRNABERT

    # Pre-tokenized mRNA sequences: UTR regions use single-letter tokens and
    # CDS regions use three-letter codon tokens, space-separated.
    sample_sequences = [
        "A T C G G A GGG CCC TTT",
        "A T C G",
        "TTT CCC GAC ATG",
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="mRNABERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
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

        model = _load_mrnabert_model(model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_sequences,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )
        return inputs
