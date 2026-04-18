# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Local DeciLM model implementation using native transformers v5 Llama modules.

The HuggingFace-hosted DeciLM code bundles transformers v4.35 internals that are
incompatible with transformers v5 and XLA devices.  This module rebuilds the model
from standard Llama components, only customising the per-layer KV head count.
"""

import copy
import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
)


def load_decilm(pretrained_model_name, *, dtype=None):
    config_path = hf_hub_download(pretrained_model_name, "config.json")
    with open(config_path) as f:
        raw = json.load(f)

    kv_per_layer = raw["num_key_value_heads_per_layer"]

    config = LlamaConfig(
        vocab_size=raw["vocab_size"],
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=kv_per_layer[0],
        max_position_embeddings=raw["max_position_embeddings"],
        rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
        rope_theta=raw.get("rope_theta", 10000.0),
    )

    model = LlamaForCausalLM(config)

    for i in range(config.num_hidden_layers):
        layer_config = copy.deepcopy(config)
        layer_config.num_key_value_heads = kv_per_layer[i]
        model.model.layers[i] = LlamaDecoderLayer(layer_config, layer_idx=i)

    index_path = hf_hub_download(pretrained_model_name, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    state_dict = {}
    for wf in set(index["weight_map"].values()):
        path = hf_hub_download(pretrained_model_name, wf)
        state_dict.update(load_file(path))

    model.load_state_dict(state_dict, strict=True, assign=True)

    if dtype is not None:
        model = model.to(dtype)

    model.eval()
    return model
