# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Loads per-layer weights from the moonshotai/Kimi-K2-Instruct HF repo,
# dequantizing on the fly:
#
# - Linear weights are stored as FP8 (e4m3fn) with F32 block scales
#   [out/128, in/128] named `weight_scale_inv`.
#   Dequantization: bf16_out = fp8_weight.float() * scale_inv[:, None, :, None]
# - LayerNorm weights, embeddings, and gate biases ship as BF16/FP32 and
#   are loaded as-is.
#
# Only the shard(s) containing the requested layers are downloaded;
# subsequent runs hit the standard huggingface_hub cache.

from __future__ import annotations

import json
import time
from typing import Dict, Iterable, List

import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from safetensors import safe_open

REPO_ID = "moonshotai/Kimi-K2-Instruct"

_FP8_BLOCK = 128


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _dequant_fp8(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """[out, in] fp8_e4m3fn + [ceil(out/128), ceil(in/128)] f32 -> [out, in] bf16.

    Handles dimensions that are not exact multiples of 128 by zero-padding
    to the next block boundary, dequantizing, then slicing back.
    """
    out_dim, in_dim = weight.shape
    out_blocks = _ceil_div(out_dim, _FP8_BLOCK)
    in_blocks = _ceil_div(in_dim, _FP8_BLOCK)
    assert scale_inv.shape == (
        out_blocks,
        in_blocks,
    ), f"fp8 scale shape mismatch: weight={weight.shape}, scale_inv={scale_inv.shape}"

    # Pad to full block boundaries if needed.
    pad_out = out_blocks * _FP8_BLOCK - out_dim
    pad_in = in_blocks * _FP8_BLOCK - in_dim
    if pad_out or pad_in:
        weight = torch.nn.functional.pad(weight, (0, pad_in, 0, pad_out))

    w = (
        weight.to(torch.float32)
        .unflatten(0, (-1, _FP8_BLOCK))
        .unflatten(-1, (-1, _FP8_BLOCK))
    )  # [bOut, 128, bIn, 128]
    s = scale_inv.to(torch.float32)[:, None, :, None]  # [bOut, 1, bIn, 1]
    result = (w * s).flatten(2, 3).flatten(0, 1).to(torch.bfloat16)

    # Slice back to original dimensions.
    if pad_out or pad_in:
        result = result[:out_dim, :in_dim]
    return result


def _find_shards_for_keys(
    weight_map: Dict[str, str], prefixes: Iterable[str]
) -> List[str]:
    prefixes = tuple(prefixes)
    return sorted({shard for k, shard in weight_map.items() if k.startswith(prefixes)})


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
    index_path = hf_hub_download(
        repo_id=REPO_ID, filename="model.safetensors.index.json"
    )
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
        shard_path = hf_hub_download(repo_id=REPO_ID, filename=shard)
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


def _dequant_paired(
    raw: Dict[str, torch.Tensor], base_prefix: str
) -> Dict[str, torch.Tensor]:
    """Walk `raw` under `base_prefix`, combining `.weight`/`.weight_scale_inv`
    pairs into bf16 tensors keyed by the trimmed local name (base_prefix
    stripped).
    """
    t0 = time.monotonic()
    out: Dict[str, torch.Tensor] = {}
    n_fp8 = 0
    n_passthrough = 0
    n_other = 0
    total_bf16_bytes = 0

    # First pass: find all .weight tensors under base_prefix.
    weights = {
        k: v
        for k, v in raw.items()
        if k.startswith(base_prefix) and k.endswith(".weight")
    }
    logger.info(
        f"[weight_loader] Dequant: {len(weights)} .weight tensors to process "
        f"(base_prefix={base_prefix!r})"
    )
    for wkey, w in weights.items():
        skey = wkey + "_scale_inv"
        local = wkey[len(base_prefix) :]  # e.g. "model.layers.0.self_attn.q_a_proj.weight"
        scale_inv = raw.get(skey)
        if scale_inv is None:
            # No scale: BF16/FP32 tensor (layernorm, embedding, gate, etc.)
            out[local] = w.to(torch.bfloat16) if w.is_floating_point() else w
            n_passthrough += 1
            logger.debug(
                f"[weight_loader]   passthrough {local}: {list(w.shape)} {w.dtype}"
            )
        else:
            # FP8 e4m3fn with block scale_inv.
            out[local] = _dequant_fp8(w, scale_inv)
            n_fp8 += 1
            logger.debug(
                f"[weight_loader]   dequant fp8 {local}: {list(w.shape)} "
                f"+ scale {list(scale_inv.shape)} -> bf16"
            )
        total_bf16_bytes += out[local].nelement() * out[local].element_size()

    # Pass through non-weight/non-scale tensors (e.g. biases) under base_prefix.
    for k, v in raw.items():
        if not k.startswith(base_prefix):
            continue
        if k.endswith(".weight") or k.endswith(".weight_scale_inv"):
            continue
        local = k[len(base_prefix) :]
        out[local] = v
        n_other += 1
        total_bf16_bytes += v.nelement() * v.element_size()
        logger.debug(f"[weight_loader]   other {local}: {list(v.shape)} {v.dtype}")

    elapsed = time.monotonic() - t0
    logger.info(
        f"[weight_loader] Dequant complete in {elapsed:.1f}s: "
        f"{n_fp8} fp8->bf16, {n_passthrough} passthrough, {n_other} other "
        f"=> {len(out)} output tensors, {total_bf16_bytes / (1024**3):.2f} GB"
    )
    return out


def load_transformer_state_dict(
    layer_ids: Iterable[int],
) -> Dict[str, torch.Tensor]:
    """Full DeepseekV3ForCausalLM state dict for the requested layer subset
    plus top-level (embed_tokens, norm, lm_head). Load with strict=False --
    non-persistent buffers and cache tensors aren't in the checkpoint.
    """
    layer_ids = sorted(set(layer_ids))
    logger.info(
        f"[weight_loader] load_transformer_state_dict: "
        f"layers={layer_ids} ({len(layer_ids)} layers)"
    )
    t_total = time.monotonic()

    prefixes: List[str] = ["model.embed_tokens.", "model.norm.", "lm_head."]
    prefixes.extend(f"model.layers.{L}." for L in layer_ids)
    raw = _load_raw_subset(prefixes)
    sd = _dequant_paired(raw, "")

    # Log summary of what we produced.
    total_params = sum(t.numel() for t in sd.values())
    total_bytes = sum(t.nelement() * t.element_size() for t in sd.values())
    dtypes = {}
    for t in sd.values():
        dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
    logger.info(
        f"[weight_loader] State dict ready: {len(sd)} keys, "
        f"{total_params / 1e9:.2f}B params, {total_bytes / (1024**3):.2f} GB, "
        f"dtypes={dtypes}, took {time.monotonic() - t_total:.1f}s total"
    )
    return sd
