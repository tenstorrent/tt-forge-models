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
    pending_layers = set(range(n_layers))
    total_shards = len(safetensor_files)
    print(
        f"[meta_loading] Loading weights from {total_shards} shard(s); "
        f"target layers: {sorted(pending_layers) if n_layers else '[]'}"
    )
    for shard_idx, path in enumerate(safetensor_files, start=1):
        print(
            f"[meta_loading] Shard {shard_idx}/{total_shards}: {os.path.basename(path)}"
        )
        chunk = safetensors_load_file(path)
        layers_in_shard: set = set()
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
            m = _LAYER_INDEX_RE.search(dst_key)
            if m is not None:
                layers_in_shard.add(int(m.group(1)))
        pending_layers -= layers_in_shard
        print(
            f"[meta_loading]   loaded layers from shard: {sorted(layers_in_shard)}; "
            f"still pending: {sorted(pending_layers)}"
        )
    return state_dict


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
    print(f"[meta_loading] Resolving shards for {repo_id} (first {n_layers} layer(s))")
    try:
        index_path = hf_hub_download(
            repo_id, "model.safetensors.index.json", revision=revision
        )
    except Exception:
        # Single-shard repo: no index.json. Fall back to model.safetensors.
        print(f"[meta_loading] No index.json; downloading single model.safetensors")
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

    sorted_shards = sorted(needed_shards)
    total = len(sorted_shards)
    print(f"[meta_loading] Need {total} shard(s) for first {n_layers} layer(s)")
    paths: List[str] = []
    for i, s in enumerate(sorted_shards, start=1):
        print(f"[meta_loading] Downloading shard {i}/{total}: {s}")
        paths.append(hf_hub_download(repo_id, s, revision=revision))
    return paths


def _rematerialize_rotary_buffers_on_cpu(model: nn.Module) -> None:
    """Rebuild each attention layer's rotary embedding on CPU after meta-load.

    The rotary buffers (``inv_freq``/``cos_cached``/``sin_cached``) are
    registered with ``persistent=False``, so they are absent from the
    checkpoint and stay on the meta device after meta-build +
    ``load_state_dict``. Any meta tensor remaining on the model trips the
    unmaterialized-meta guard (and would otherwise die on the first forward with
    ``Cannot copy out of meta tensor``).

    Re-running ``_init_rope`` under ``torch.device("cpu")`` reconstructs each
    rotary module with all three buffers materialized on CPU. This only exists
    to materialize the buffers: ``DeepseekV3RotaryEmbedding.__init__`` resets
    ``max_seq_len_cached`` to ``None``, so the first forward recomputes the
    cache on the real input device — ``apply_rotary_pos_emb`` therefore never
    reads these CPU values. (Do not pre-populate ``max_seq_len_cached`` here:
    that would skip the forward-time recompute and leak CPU tensors into an
    otherwise on-device attention.)
    """
    with torch.device("cpu"):
        for module in model.modules():
            if hasattr(module, "_init_rope"):
                module._init_rope()


def load_model_from_checkpoint(
    model_factory: Callable[[], nn.Module],
    checkpoint: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    n_layers: int,
    *,
    rename_key: Optional[Callable[[str], Optional[str]]] = None,
    revision: Optional[str] = None,
    layer_pattern: "re.Pattern[str]" = _DEFAULT_LAYER_PATTERN,
    drop_key_prefixes: Iterable[str] = ("mtp.",),
    drop_key_substrings: Iterable[str] = (".mtp.",),
) -> nn.Module:
    """Build a model on meta and populate it from a safetensors checkpoint.

    Args:
        model_factory: Zero-arg callable that constructs and returns the model.
            Invoked under ``torch.device("meta")`` so it must not require real
            tensor storage during ``__init__``.
        checkpoint: One of:
            * an HF Hub repo id (e.g. ``"deepseek-ai/DeepSeek-V3.1"``): only the
              shards holding the first ``n_layers`` are resolved and downloaded
              via ``resolve_hf_shards_for_layers``;
            * a local directory containing one or more ``*.safetensors`` shards
              (all are loaded);
            * an explicit sequence of safetensors file paths.
        n_layers: Number of decoder layers to load (keys matching
            ``layers.{idx}.`` with ``idx >= n_layers`` are dropped). Non-layer
            keys (embeddings, final norm, lm head, etc.) are always kept.
        rename_key: Optional callback applied to each checkpoint key before
            assignment. Returning ``None`` drops the tensor (useful for skipping
            quantization-scale auxiliaries like ``*.weight_scale_inv``).
        revision, layer_pattern, drop_key_prefixes, drop_key_substrings: Passed
            through to ``resolve_hf_shards_for_layers`` when ``checkpoint`` is an
            HF repo id; ignored for local paths.

    Returns:
        The model with weights assigned from the checkpoint.
    """
    with torch.device("meta"):
        model = model_factory()

    if isinstance(checkpoint, (str, os.PathLike)) and not os.path.isdir(checkpoint):
        # Treat as an HF Hub repo id: resolve only the shards holding the
        # first ``n_layers`` and download them.
        safetensor_files = resolve_hf_shards_for_layers(
            str(checkpoint),
            n_layers,
            revision=revision,
            layer_pattern=layer_pattern,
            drop_key_prefixes=drop_key_prefixes,
            drop_key_substrings=drop_key_substrings,
        )
    else:
        safetensor_files = _resolve_safetensor_files(checkpoint)

    state_dict = _load_filtered_state_dict(
        safetensor_files, n_layers, rename_key=rename_key
    )
    model.load_state_dict(state_dict, strict=False, assign=True)

    # Non-persistent rotary buffers are not in the checkpoint, so they remain on
    # the meta device after the assign above; re-materialize them on CPU.
    _rematerialize_rotary_buffers_on_cpu(model)

    return model
