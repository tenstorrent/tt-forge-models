# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared HuggingFace model/tokenizer loading helpers for forge model loaders."""

from __future__ import annotations

from typing import Optional

import torch

from .transformers_compat import apply_transformers_compat_patches, load_causal_lm, load_tokenizer


def load_causal_lm_from_variant(
    model_id: str,
    *,
    dtype_override: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
    **kwargs,
):
    """Standard causal LM load path used by many forge model loaders."""
    apply_transformers_compat_patches()
    model_kwargs = dict(kwargs)
    if dtype_override is not None:
        model_kwargs.setdefault("torch_dtype", dtype_override)
    model = load_causal_lm(model_id, trust_remote_code=trust_remote_code, **model_kwargs)
    model.eval()
    return model


def load_tokenizer_from_variant(
    model_id: str,
    *,
    trust_remote_code: bool = False,
    pad_to_eos: bool = True,
    **kwargs,
):
    """Standard tokenizer load path; never forwards torch_dtype to the tokenizer."""
    apply_transformers_compat_patches()
    tokenizer = load_tokenizer(model_id, trust_remote_code=trust_remote_code, **kwargs)
    if pad_to_eos and tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
