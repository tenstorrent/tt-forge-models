# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-4 MoE specific helpers for meta-building the model and loading its
weights from HF safetensors shards."""

import json
import re

import torch
from huggingface_hub import hf_hub_download

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _resolve_hf_shards_for_layers(repo_id, n_layers, *, revision=None):
    """Return local paths of safetensors shards holding the first ``n_layers``
    decoder layers from ``repo_id``.

    Parses ``model.safetensors.index.json``, filters its ``weight_map`` to
    keys whose layer index is ``< n_layers`` (or that don't carry a layer
    index — embeddings, final norm, lm head), drops MTP / next-N keys, and
    ``hf_hub_download``-s only the unique shards needed. Falls back to a single
    ``model.safetensors`` download for single-shard repos.
    """
    print(f"[meta_loading] Resolving shards for {repo_id} (first {n_layers} layer(s))")
    try:
        index_path = hf_hub_download(
            repo_id, "model.safetensors.index.json", revision=revision
        )
    except Exception:
        # Single-shard repo: no index.json. Fall back to model.safetensors.
        print("[meta_loading] No index.json; downloading single model.safetensors")
        return [hf_hub_download(repo_id, "model.safetensors", revision=revision)]

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    needed_shards = set()
    for ckpt_key, shard_name in weight_map.items():
        m = _LAYER_INDEX_RE.search(ckpt_key)
        if m and int(m.group(1)) >= n_layers:
            continue
        if ckpt_key.startswith("mtp.") or ".mtp." in ckpt_key:
            continue
        needed_shards.add(shard_name)

    sorted_shards = sorted(needed_shards)
    total = len(sorted_shards)
    print(f"[meta_loading] Need {total} shard(s) for first {n_layers} layer(s)")
    paths = []
    for i, s in enumerate(sorted_shards, start=1):
        print(f"[meta_loading] Downloading shard {i}/{total}: {s}")
        paths.append(hf_hub_download(repo_id, s, revision=revision))
    return paths


def _build_glm4_config(pretrained_model_name, num_layers):
    from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig

    # Strict config prevents unknown fields (e.g. _experts_implementation) from
    # being stored, which would cause a KeyError in ExpertsInterface dispatch.
    config = Glm4MoeConfig.from_pretrained(pretrained_model_name)
    config.num_hidden_layers = num_layers
    config._attn_implementation = "eager"
    if hasattr(config, "num_nextn_predict_layers"):
        config.num_nextn_predict_layers = 0
    return config


def _load_glm4_state_dict(pretrained_model_name, n_layers, n_experts):
    """Load the first n_layers from HF shards, fusing per-expert weights into 3-D tensors.

    HF stores experts as model.layers.L.mlp.experts.J.{gate,up,down}_proj.weight
    but Glm4MoeNaiveMoe expects gate_up_proj [E, 2*inter, H] and down_proj [E, H, inter].
    """
    from safetensors.torch import load_file as safetensors_load_file

    expert_re = re.compile(
        r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)"
        r"\.(gate_proj|up_proj|down_proj)\.weight$"
    )
    layer_re = re.compile(r"(?:^|\.)layers\.(\d+)\.")

    expert_tensors = {}
    state_dict = {}
    for shard_path in _resolve_hf_shards_for_layers(pretrained_model_name, n_layers):
        for k, t in safetensors_load_file(shard_path, device="cpu").items():
            m = layer_re.search(k)
            if m and int(m.group(1)) >= n_layers:
                continue
            t = t.to(torch.bfloat16)
            m = expert_re.match(k)
            if m:
                expert_tensors[(int(m.group(1)), int(m.group(2)), m.group(3))] = t
            else:
                state_dict[k] = t

    for layer_idx in sorted({k[0] for k in expert_tensors}):
        gate = torch.stack(
            [expert_tensors[(layer_idx, j, "gate_proj")] for j in range(n_experts)]
        )
        up = torch.stack(
            [expert_tensors[(layer_idx, j, "up_proj")] for j in range(n_experts)]
        )
        down = torch.stack(
            [expert_tensors[(layer_idx, j, "down_proj")] for j in range(n_experts)]
        )
        state_dict[f"model.layers.{layer_idx}.mlp.experts.gate_up_proj"] = torch.cat(
            [gate, up], dim=1
        ).contiguous()
        state_dict[
            f"model.layers.{layer_idx}.mlp.experts.down_proj"
        ] = down.contiguous()

    return state_dict


def _restore_glm4_remaining_meta_tensors(model, config):
    """Apply post-load fixups required after meta-device construction."""
    from transformers.models.glm4_moe.modeling_glm4_moe import (
        Glm4MoeRotaryEmbedding,
        Glm4MoeTopkRouter,
    )

    # inv_freq is not persisted in checkpoints; re-initialize on CPU.
    model.model.rotary_emb = Glm4MoeRotaryEmbedding(config, device="cpu")

    # bf16 rounding of e_score_correction_bias flips top-k expert selections.
    for module in model.modules():
        if (
            isinstance(module, Glm4MoeTopkRouter)
            and module.e_score_correction_bias.dtype != torch.float32
        ):
            module.e_score_correction_bias = module.e_score_correction_bias.to(
                torch.float32
            )


def load_model_for_num_layers(pretrained_model_name, num_layers):
    """Build a GLM-4 MoE model on the meta device and populate the first
    ``num_layers`` decoder layers from the HF safetensors checkpoint.

    Args:
        pretrained_model_name: HF Hub repo id, e.g. ``"zai-org/GLM-4.7"``.
        num_layers: Number of decoder layers to construct and load.

    Returns:
        The populated ``Glm4MoeForCausalLM`` instance.
    """
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeForCausalLM

    config = _build_glm4_config(pretrained_model_name, num_layers)

    with torch.device("meta"):
        model = Glm4MoeForCausalLM(config)

    state_dict = _load_glm4_state_dict(
        pretrained_model_name, num_layers, config.n_routed_experts
    )
    model.load_state_dict(state_dict, strict=False, assign=True)

    _restore_glm4_remaining_meta_tensors(model, config)

    return model
