# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shims for HuggingFace Transformers 5.x in forge model tests.

Transformers 5.5.1 (the repo default in venv/requirements-dev.txt) changed
``PreTrainedModel.initialize_weights`` to call ``smart_apply(fn, is_remote_code)``.
Remote modeling code and some native configs still expose the older 2-argument
``smart_apply`` signature, which raises ``TypeError`` during weight init.

Passing a loaded ``PreTrainedConfig`` instance as the first positional argument
to ``from_pretrained`` is also invalid: Transformers coerces it with ``str()``
which produces strings like ``Qwen2Config { ... }`` and HuggingFace Hub rejects
them as repo ids.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional


def _config_repo_id(config: Any) -> Optional[str]:
    for attr in ("name_or_path", "_name_or_path"):
        value = getattr(config, attr, None)
        if value:
            return str(value)
    return None


def _coerce_pretrained_path(pretrained_model_name_or_path: Any, kwargs: dict) -> Any:
    try:
        from transformers.configuration_utils import PreTrainedConfig
    except ImportError:
        return pretrained_model_name_or_path

    if not isinstance(pretrained_model_name_or_path, PreTrainedConfig):
        return pretrained_model_name_or_path

    config = pretrained_model_name_or_path
    repo_id = _config_repo_id(config)
    if not repo_id:
        raise ValueError(
            "A PreTrainedConfig object was passed as the first argument to "
            "from_pretrained, but the config has no name_or_path set. Pass the "
            "HuggingFace repo id string instead, e.g. "
            "AutoModelForCausalLM.from_pretrained('org/model', config=config)."
        )

    kwargs.setdefault("config", config)
    return repo_id


def _patch_from_pretrained(classmethod_fn: Callable) -> Callable:
    @functools.wraps(classmethod_fn)
    def wrapped(cls, pretrained_model_name_or_path, *args, **kwargs):
        pretrained_model_name_or_path = _coerce_pretrained_path(
            pretrained_model_name_or_path, kwargs
        )
        return classmethod_fn(cls, pretrained_model_name_or_path, *args, **kwargs)

    return wrapped


def _install_compatible_smart_apply() -> None:
    import torch
    from transformers.modeling_utils import PreTrainedModel

    def _call_init_fn(module, fn, is_remote_code):
        try:
            return fn(module, is_remote_code)
        except TypeError:
            try:
                return fn(module)
            except TypeError:
                raise

    def compatible_smart_apply(module, fn, is_remote_code):
        for child in module.children():
            if isinstance(child, PreTrainedModel):
                compatible_smart_apply(child, child._initialize_weights, is_remote_code)
            else:
                compatible_smart_apply(child, fn, is_remote_code)
        _call_init_fn(module, fn, is_remote_code)
        return module

    torch.nn.Module.smart_apply = compatible_smart_apply
    torch.nn.Module._tt_forge_smart_apply_patched = True


def apply_transformers_compat_patches() -> None:
    """Install Transformers compatibility patches for the loaded transformers build."""
    try:
        from transformers.configuration_utils import PreTrainedConfig
        from transformers.modeling_utils import PreTrainedModel
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except ImportError:
        return

    _install_compatible_smart_apply()

    if not getattr(PreTrainedModel, "_tt_forge_compat_patched", False):
        PreTrainedModel.from_pretrained = classmethod(
            _patch_from_pretrained(PreTrainedModel.from_pretrained.__func__)
        )
        PreTrainedModel._tt_forge_compat_patched = True

    if not getattr(PreTrainedTokenizerBase, "_tt_forge_compat_patched", False):
        PreTrainedTokenizerBase.from_pretrained = classmethod(
            _patch_from_pretrained(PreTrainedTokenizerBase.from_pretrained.__func__)
        )
        PreTrainedTokenizerBase._tt_forge_compat_patched = True

    if not getattr(PreTrainedConfig, "_tt_forge_compat_patched", False):
        PreTrainedConfig.from_pretrained = classmethod(
            _patch_from_pretrained(PreTrainedConfig.from_pretrained.__func__)
        )
        PreTrainedConfig._tt_forge_compat_patched = True


def load_tokenizer(model_id: str, *, trust_remote_code: bool = False, **kwargs):
    """Load a tokenizer without passing invalid kwargs such as torch_dtype."""
    apply_transformers_compat_patches()
    from transformers import AutoTokenizer

    tokenizer_kwargs = dict(kwargs)
    tokenizer_kwargs.pop("torch_dtype", None)
    tokenizer_kwargs.pop("dtype", None)
    if trust_remote_code:
        tokenizer_kwargs.setdefault("trust_remote_code", True)
    return AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)


def load_causal_lm(model_id: str, *, trust_remote_code: bool = False, **kwargs):
    """Load a causal LM using a repo id string and optional trust_remote_code."""
    apply_transformers_compat_patches()
    from transformers import AutoModelForCausalLM

    model_kwargs = dict(kwargs)
    if trust_remote_code:
        model_kwargs.setdefault("trust_remote_code", True)
    return AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
