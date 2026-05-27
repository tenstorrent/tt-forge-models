# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for meta-building a torch model and populating its weights
from a safetensors checkpoint.

The util is decoder-only: layer keys are matched against ``layers.<N>.``.
Encoder/decoder architectures (T5, BART, Whisper) need a different
``layer_pattern`` and are not supported by the default helpers below.
"""
import json
import os
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safetensors_load_file
from torch import nn

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _tensor_belongs_to_first_n_layers(key: str, n_layers: int) -> bool:
    """True if ``key`` is either non-layer-scoped or names a layer < ``n_layers``."""
    match = _LAYER_INDEX_RE.search(key)
    if match is None:
        return True
    return int(match.group(1)) < n_layers


def _resolve_safetensor_files(
    checkpoint: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
) -> List[str]:
    """Normalize ``checkpoint`` (directory or sequence of file paths) to a
    sorted list of safetensors file paths."""
    if isinstance(checkpoint, (str, os.PathLike)):
        return [
            os.path.join(checkpoint, f)
            for f in sorted(os.listdir(checkpoint))
            if f.endswith(".safetensors")
        ]
    return [str(p) for p in checkpoint]


def _load_filtered_state_dict(
    safetensor_files: Sequence[str],
    n_layers: int,
    rename_key: Optional[Callable[[str], Optional[str]]] = None,
) -> Dict[str, torch.Tensor]:
    """Merge safetensors files, keeping only tensors that belong to the first
    ``n_layers``. Optional ``rename_key`` translates source keys to the model's
    expected naming (returning ``None`` drops the tensor)."""
    state_dict: Dict[str, torch.Tensor] = {}
    for path in safetensor_files:
        chunk = safetensors_load_file(path)
        for src_key, tensor in chunk.items():
            if not _tensor_belongs_to_first_n_layers(src_key, n_layers):
                continue
            dst_key = src_key
            if rename_key is not None:
                renamed = rename_key(src_key)
                if renamed is None:
                    continue
                dst_key = renamed
                if not _tensor_belongs_to_first_n_layers(dst_key, n_layers):
                    continue
            state_dict[dst_key] = tensor
    return state_dict


def _collect_meta_tensors(model: nn.Module) -> List[str]:
    """Return fully-qualified names of any parameters or buffers still on meta."""
    meta_names: List[str] = []
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            meta_names.append(name)
    for name, buf in model.named_buffers():
        if buf is not None and buf.device.type == "meta":
            meta_names.append(name)
    return meta_names


class UnmaterializedMetaTensorError(RuntimeError):
    """Raised when meta tensors remain on the model after checkpoint load.

    These are typically non-persistent buffers (``register_buffer(...,
    persistent=False)``) or parameters whose checkpoint keys were missing or
    filtered out — neither path materializes CPU storage, so the first forward
    pass dies with ``Cannot copy out of meta tensor``.

    The loader for the affected model must apply a fixup after
    ``load_meta_model_from_checkpoint`` returns. Typical fixups:

    * Replace a submodule that computes a buffer in its ``__init__`` (e.g. HF
      rotary embedding ``inv_freq``) with a fresh instance constructed on CPU::

          model.submod = SomeModule(config, device="cpu")

    * Re-allocate a non-persistent cache buffer on CPU::

          module.cache = torch.zeros(..., device="cpu")

    * Recompute a derived tensor from config and assign it::

          model.derived = compute_derived(config)
    """


def load_meta_model_from_checkpoint(
    model_factory: Callable[[], nn.Module],
    checkpoint: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    n_layers: int,
    *,
    rename_key: Optional[Callable[[str], Optional[str]]] = None,
    check_unmaterialized_meta: bool = True,
) -> nn.Module:
    """Build a model on meta and populate it from a safetensors checkpoint.

    Args:
        model_factory: Zero-arg callable that constructs and returns the model.
            Invoked under ``torch.device("meta")`` so it must not require real
            tensor storage during ``__init__``.
        checkpoint: Either a directory containing one or more
            ``*.safetensors`` shards (all are loaded) or an explicit sequence
            of safetensors file paths. Callers that want to load only a subset
            of HF shards (e.g. just the ones holding the first ``n_layers``)
            should resolve those paths via ``hf_hub_download`` (see
            ``resolve_hf_shards_for_layers`` below) and pass them in.
        n_layers: Number of decoder layers to load (keys matching
            ``layers.{idx}.`` with ``idx >= n_layers`` are dropped). Non-layer
            keys (embeddings, final norm, lm head, etc.) are always kept.
        rename_key: Optional callback applied to each checkpoint key before
            assignment. Returning ``None`` drops the tensor (useful for skipping
            quantization-scale auxiliaries like ``*.weight_scale_inv``).
        check_unmaterialized_meta: If True (default), after assignment scan
            the model for any parameters/buffers still on the meta device and
            raise :class:`UnmaterializedMetaTensorError` listing them. The
            calling loader is expected to apply its model-specific fixup
            *before* calling this util (e.g. by wrapping the call in a helper
            that runs the fixup and re-checks) or to pass
            ``check_unmaterialized_meta=False`` and handle materialization
            itself. Set to False only when the caller is doing its own
            post-load rematerialization pass.

    Returns:
        The model with weights assigned from the checkpoint.

    Raises:
        UnmaterializedMetaTensorError: If meta tensors remain after load and
            ``check_unmaterialized_meta`` is True.
    """
    with torch.device("meta"):
        model = model_factory()

    safetensor_files = _resolve_safetensor_files(checkpoint)
    state_dict = _load_filtered_state_dict(
        safetensor_files, n_layers, rename_key=rename_key
    )
    model.load_state_dict(state_dict, strict=False, assign=True)

    if check_unmaterialized_meta:
        meta_names = _collect_meta_tensors(model)
        if meta_names:
            preview = "\n  - ".join(meta_names[:20])
            extra = (
                f"\n  ... and {len(meta_names) - 20} more"
                if len(meta_names) > 20
                else ""
            )
            raise UnmaterializedMetaTensorError(
                f"{len(meta_names)} tensor(s) remained on the meta device after "
                f"load_state_dict. The loader for this model must implement a "
                f"fixup that re-materializes these on CPU (typically by "
                f"rebuilding the owning submodule with device='cpu', or by "
                f"reassigning the buffer with a freshly-computed CPU tensor). "
                f"See UnmaterializedMetaTensorError docstring for patterns. "
                f"Offending tensors:\n  - {preview}{extra}\n"
                f"To opt out of this check (e.g. you handle materialization "
                f"yourself), pass check_unmaterialized_meta=False."
            )

    return model


_DEFAULT_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def resolve_hf_shards_for_layers(
    repo_id: str,
    n_layers: int,
    *,
    revision: Optional[str] = None,
    layer_pattern: "re.Pattern[str]" = _DEFAULT_LAYER_PATTERN,
    drop_key_prefixes: Iterable[str] = ("mtp.",),
    drop_key_substrings: Iterable[str] = (".mtp.",),
) -> List[str]:
    """Return local paths of safetensors shards holding the first ``n_layers``
    decoder layers from ``repo_id``.

    Parses ``model.safetensors.index.json``, filters its ``weight_map`` to
    keys whose layer index is ``< n_layers`` (or that don't carry a layer
    index — embeddings, final norm, lm head), drops MTP / next-N keys, and
    ``hf_hub_download``-s only the unique shards needed.

    If the repo has no ``model.safetensors.index.json`` (single-shard repo),
    falls back to downloading the single ``model.safetensors`` file.

    Args:
        repo_id: HF Hub repo, e.g. ``"deepseek-ai/DeepSeek-V3.1"``.
        n_layers: Keys naming layer indices ``>= n_layers`` are excluded
            when picking which shards to fetch.
        revision: Optional HF revision (branch/tag/commit).
        layer_pattern: Compiled regex with one numeric capture group for the
            layer index. Default matches the Llama/Qwen/Mistral/DeepSeek
            ``layers.<N>.`` convention.
        drop_key_prefixes / drop_key_substrings: Keys with any of these
            prefixes / substrings are ignored when computing the needed
            shard set (MTP heads are a common case).

    Returns:
        Sorted list of local safetensors file paths.
    """
    try:
        index_path = hf_hub_download(
            repo_id, "model.safetensors.index.json", revision=revision
        )
    except Exception:
        # Single-shard repo: no index.json. Fall back to model.safetensors.
        return [hf_hub_download(repo_id, "model.safetensors", revision=revision)]

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    needed_shards = set()
    for ckpt_key, shard_name in weight_map.items():
        m = layer_pattern.search(ckpt_key)
        if m and int(m.group(1)) >= n_layers:
            continue
        if any(ckpt_key.startswith(p) for p in drop_key_prefixes):
            continue
        if any(s in ckpt_key for s in drop_key_substrings):
            continue
        needed_shards.add(shard_name)

    return [
        hf_hub_download(repo_id, s, revision=revision) for s in sorted(needed_shards)
    ]
