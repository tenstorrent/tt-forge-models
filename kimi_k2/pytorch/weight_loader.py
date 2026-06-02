# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Loads per-layer BF16 weights for Kimi K2 from the unsloth/Kimi-K2-Base-BF16
# HF repo. HuggingFace handles all downloading and caching (under HF_HOME); only
# the safetensors shard(s) containing the requested layers are fetched, which
# bounds peak host memory to roughly one layer's worth of weights.
#
# The unsloth BF16 reupload is already dequantized, so there is no FP8
# block-scale handling here -- weights load straight through.

from __future__ import annotations

import os

# Default HF cache location. Set before importing huggingface_hub so (ideally)
# its cache constants (HF_HUB_CACHE = $HF_HOME/hub) pick it up. Export HF_HOME to
# override. NOTE: if huggingface_hub was already imported elsewhere before this
# runs, its constants freeze to the prior value (e.g. ~/.cache/huggingface), so
# we additionally pass an explicit cache_dir on every download below to be
# import-order-proof.
DEFAULT_HF_HOME = "/mnt/models/users/jzx"
os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)

import json
import time
from typing import Dict, Iterable, List

import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from safetensors import safe_open

REPO_ID = "unsloth/Kimi-K2-Base-BF16"

_LOGGED_CACHE_DIAGNOSTICS = False


def _log_cache_diagnostics() -> None:
    """Log where HF will actually read/write the checkpoint, once per process.

    ``HF_HOME`` sets the base, but the hub blob cache and the Xet chunk cache can
    be redirected independently via ``HF_HUB_CACHE`` / ``HUGGINGFACE_HUB_CACHE``
    and ``HF_XET_CACHE``. This logs both the relevant env vars and the resolved
    huggingface_hub constants so it's obvious where the weights land (e.g. only
    a ``xet/`` dir showing up under HF_HOME while blobs go elsewhere).
    """
    global _LOGGED_CACHE_DIAGNOSTICS
    if _LOGGED_CACHE_DIAGNOSTICS:
        return
    _LOGGED_CACHE_DIAGNOSTICS = True

    env_keys = [
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_XET_CACHE",
        "HF_HUB_DISABLE_XET",
        "HF_HUB_OFFLINE",
    ]
    logger.info(f"[weight_loader] HF cache diagnostics for repo {REPO_ID!r}:")
    for k in env_keys:
        logger.info(f"[weight_loader]   env {k}={os.environ.get(k, '<unset>')}")

    try:
        from huggingface_hub import constants as hf_constants

        for name in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
            logger.info(
                f"[weight_loader]   huggingface_hub.constants.{name}="
                f"{getattr(hf_constants, name, '<missing>')}"
            )
    except Exception as e:  # pragma: no cover - purely diagnostic
        logger.warning(f"[weight_loader] Could not read huggingface_hub constants: {e}")

    logger.info(
        f"[weight_loader]   effective cache_dir passed to hf_hub_download="
        f"{_hub_cache_dir()}"
    )


def _hub_cache_dir() -> str:
    """Resolve the hub blob cache dir from the *current* environment, rather than
    relying on huggingface_hub's import-time constants (which freeze to whatever
    HF_HOME was when the library was first imported -- often ~/.cache/huggingface
    if it was imported before our setdefault above ran). Respects an explicit
    HF_HUB_CACHE / HUGGINGFACE_HUB_CACHE override if set."""
    explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get(
        "HUGGINGFACE_HUB_CACHE"
    )
    if explicit:
        return explicit
    return os.path.join(os.environ.get("HF_HOME", DEFAULT_HF_HOME), "hub")


def _find_shards_for_keys(
    weight_map: Dict[str, str], prefixes: Iterable[str]
) -> List[str]:
    prefixes = tuple(prefixes)
    return sorted({shard for k, shard in weight_map.items() if k.startswith(prefixes)})


def _resolve_file(filename: str) -> str:
    """Download (or fetch from the HF cache) ``filename`` from ``REPO_ID`` and
    return its local path. The hub blob cache dir is passed explicitly so the
    location is import-order-proof (see ``_hub_cache_dir``)."""
    _log_cache_diagnostics()
    path = hf_hub_download(REPO_ID, filename, cache_dir=_hub_cache_dir())
    # The returned path is a symlink under snapshots/<commit>/; resolve it to the
    # real blob so the on-disk location of the actual bytes is unambiguous.
    real = os.path.realpath(path)
    logger.info(
        f"[weight_loader] Resolved {filename!r} -> {path}"
        + (f" (blob: {real})" if real != path else "")
    )
    return path


def _load_raw_subset(
    prefixes: Iterable[str],
) -> Dict[str, torch.Tensor]:
    """Download relevant shards and return tensors whose keys match any prefix.
    Keys are returned verbatim (with their original checkpoint prefix)."""
    prefixes = list(prefixes)
    logger.info(f"[weight_loader] Loading raw tensors for {len(prefixes)} prefixes")
    for p in prefixes:
        logger.debug(f"[weight_loader]   prefix: {p}")

    t0 = time.monotonic()
    index_path = _resolve_file("model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    logger.info(
        f"[weight_loader] Index loaded: {len(weight_map)} keys in weight_map "
        f"({time.monotonic() - t0:.1f}s)"
    )

    shard_names = _find_shards_for_keys(weight_map, prefixes)
    if not shard_names:
        raise RuntimeError(f"No shards found for prefixes: {list(prefixes)}")
    logger.info(f"[weight_loader] Need {len(shard_names)} shard file(s): {shard_names}")

    raw: Dict[str, torch.Tensor] = {}
    prefix_tuple = tuple(prefixes)
    total_bytes = 0
    for i, shard in enumerate(shard_names):
        t_shard = time.monotonic()
        shard_path = _resolve_file(shard)
        t_download = time.monotonic() - t_shard
        shard_keys = 0
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix_tuple):
                    tensor = f.get_tensor(key)
                    total_bytes += tensor.nelement() * tensor.element_size()
                    raw[key] = tensor
                    shard_keys += 1
        logger.info(
            f"[weight_loader] Shard {i + 1}/{len(shard_names)} {shard}: "
            f"extracted {shard_keys} tensors (download+open {t_download:.1f}s)"
        )
    logger.info(
        f"[weight_loader] Raw load complete: {len(raw)} tensors, "
        f"{total_bytes / (1024**3):.2f} GB total"
    )
    return raw


def _is_loadable_key(key: str) -> bool:
    """Filter out checkpoint keys that are NOT real, persistent parameters.

    - ``*.rotary_emb.*`` (e.g. ``inv_freq``): RoPE caches registered as
      NON-persistent buffers. They are absent from the module's ``state_dict``
      (so ``load_state_dict`` reports them as *unexpected*) and are recomputed
      deterministically from config at runtime, so the checkpoint copies are
      both unloadable and unnecessary. See note in ``load_block_state_dict``.
    """
    if ".rotary_emb." in key:
        return False
    return True


def load_top_level_state_dict() -> Dict[str, torch.Tensor]:
    """Top-level (non-layer) weights only: embed_tokens, final norm, lm_head.

    Keys are returned with their full checkpoint names (e.g.
    ``model.embed_tokens.weight``, ``model.norm.weight``, ``lm_head.weight``)
    so the dict can be loaded straight into a ``DeepseekV3ForCausalLM`` via
    ``model.load_state_dict(sd, strict=False)``. Intended for streaming load,
    where the transformer layers are loaded one at a time afterwards.
    """
    prefixes = ["model.embed_tokens.", "model.norm.", "lm_head."]
    raw = _load_raw_subset(prefixes)
    sd: Dict[str, torch.Tensor] = {
        k: v.to(torch.bfloat16) if v.is_floating_point() else v
        for k, v in raw.items()
        if _is_loadable_key(k)
    }
    logger.info(f"[weight_loader] Top-level state dict ready: {len(sd)} keys")
    return sd


def load_block_state_dict(layer_idx: int) -> Dict[str, torch.Tensor]:
    """State dict for a single transformer layer, keyed RELATIVE to the
    ``DeepseekV3DecoderLayer`` (the ``model.layers.{idx}.`` prefix is
    stripped), so it can be loaded with
    ``layer.load_state_dict(sd, strict=False)``.

    Only the shard file(s) containing this layer are read, which bounds peak
    host memory to roughly one layer's worth of weights -- the core of the
    streaming load strategy.

    Non-persistent RoPE buffers (``self_attn.rotary_emb.inv_freq`` etc.) are
    dropped via ``_is_loadable_key``: they are not part of the layer's
    ``state_dict`` and would surface as *unexpected* keys, and Kimi K2's YaRN
    rotary recomputes them from config on the first forward regardless.
    """
    prefix = f"model.layers.{layer_idx}."
    raw = _load_raw_subset([prefix])
    sd: Dict[str, torch.Tensor] = {
        k[len(prefix) :]: v.to(torch.bfloat16) if v.is_floating_point() else v
        for k, v in raw.items()
        if _is_loadable_key(k)
    }
    total_bytes = sum(t.nelement() * t.element_size() for t in sd.values())
    logger.info(
        f"[weight_loader] Block {layer_idx} state dict ready: {len(sd)} keys, "
        f"{total_bytes / (1024**3):.2f} GB"
    )
    return sd


def load_transformer_state_dict(
    layer_ids: Iterable[int],
) -> Dict[str, torch.Tensor]:
    """Full DeepseekV3ForCausalLM state dict for the requested layer subset
    plus top-level (embed_tokens, norm, lm_head). Load with strict=False --
    non-persistent buffers and cache tensors aren't in the checkpoint.

    Loads BF16 weights directly from the HF safetensors.
    """
    layer_ids = sorted(set(layer_ids))
    logger.info(
        f"[weight_loader] load_transformer_state_dict: "
        f"layers={layer_ids} ({len(layer_ids)} layers)"
    )
    t_total = time.monotonic()

    prefixes = ["model.embed_tokens.", "model.norm.", "lm_head."]
    for L in layer_ids:
        prefixes.append(f"model.layers.{L}.")

    raw = _load_raw_subset(prefixes)

    sd: Dict[str, torch.Tensor] = {
        k: v.to(torch.bfloat16) if v.is_floating_point() else v
        for k, v in raw.items()
        if _is_loadable_key(k)
    }

    total_params = sum(t.numel() for t in sd.values())
    total_bytes = sum(t.nelement() * t.element_size() for t in sd.values())
    logger.info(
        f"[weight_loader] State dict ready: {len(sd)} keys, "
        f"{total_params / 1e9:.2f}B params, {total_bytes / (1024**3):.2f} GB, "
        f"took {time.monotonic() - t_total:.1f}s total"
    )
    return sd
