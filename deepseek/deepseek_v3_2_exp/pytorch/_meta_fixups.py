# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Loader-specific helpers for porting an HF DeepSeek-V3.x checkpoint onto
the locally-modified custom ``Transformer`` and re-materializing the
non-persistent buffers that meta-construction leaves on the meta device.
"""
import re

import scipy.linalg
import torch

from .src.modified_model import precompute_freqs_cis


def _rename_hf_key(ckpt_key, n_dense_layers=1):
    """Rename a HuggingFace checkpoint key to match modified_model.py naming.

    Returns ``None`` to drop a tensor (FP8 scale-inv auxiliaries; per-expert
    MoE weights on layers that are dense, where the HF dense MLP keys
    co-exist with MoE keys above ``n_dense_layers``).
    """
    key = ckpt_key
    if key.startswith("model."):
        key = key[len("model.") :]
    if "weight_scale_inv" in key:
        return None
    key = key.replace("lm_head.", "head.")
    key = key.replace("embed_tokens.", "embed.")
    key = re.sub(r"(layers\.\d+\.)input_layernorm\.", r"\1attn_norm.", key)
    key = re.sub(r"(layers\.\d+\.)post_attention_layernorm\.", r"\1ffn_norm.", key)
    key = key.replace("self_attn.indexer.", "attn.indexer.")
    key = key.replace("self_attn.q_a_proj.", "attn.wq_a.")
    key = key.replace("self_attn.q_b_proj.", "attn.wq_b.")
    key = key.replace("self_attn.q_a_layernorm.", "attn.q_norm.")
    key = key.replace("self_attn.kv_a_proj_with_mqa.", "attn.wkv_a.")
    key = key.replace("self_attn.kv_b_proj.", "attn.wkv_b.")
    key = key.replace("self_attn.kv_a_layernorm.", "attn.kv_norm.")
    key = key.replace("self_attn.o_proj.", "attn.wo.")
    key = re.sub(r"mlp\.experts\.(\d+)\.gate_proj\.", r"ffn.experts.\1.w1.", key)
    key = re.sub(r"mlp\.experts\.(\d+)\.down_proj\.", r"ffn.experts.\1.w2.", key)
    key = re.sub(r"mlp\.experts\.(\d+)\.up_proj\.", r"ffn.experts.\1.w3.", key)
    key = key.replace("mlp.shared_experts.gate_proj.", "ffn.shared_experts.w1.")
    key = key.replace("mlp.shared_experts.down_proj.", "ffn.shared_experts.w2.")
    key = key.replace("mlp.shared_experts.up_proj.", "ffn.shared_experts.w3.")
    key = key.replace("mlp.gate.e_score_correction_bias", "mlp.gate.bias")
    key = key.replace("mlp.gate.", "ffn.gate.")
    layer_m = re.match(r"layers\.(\d+)\.", key)
    if layer_m:
        layer_id = int(layer_m.group(1))
        if layer_id < n_dense_layers:
            key = key.replace("mlp.gate_proj.", "ffn.w1.")
            key = key.replace("mlp.down_proj.", "ffn.w2.")
            key = key.replace("mlp.up_proj.", "ffn.w3.")
        elif (
            "mlp.gate_proj." in key or "mlp.down_proj." in key or "mlp.up_proj." in key
        ):
            return None
    return key


def _fix_meta_buffers(model, args):
    """Replace meta-device buffers with properly-computed CPU tensors.

    After meta construction + ``load_state_dict(assign=True)``, non-persistent
    buffers remain on meta. Recompute them on CPU.
    """
    freqs_cis_complex = precompute_freqs_cis(args)
    model.freqs_cis = torch.view_as_real(freqs_cis_complex)
    hadamard = torch.tensor(
        scipy.linalg.hadamard(args.index_head_dim), dtype=torch.bfloat16
    ) * (args.index_head_dim**-0.5)
    model.hadamard_matrix = hadamard
    # Re-materialize the MLA cache layers on CPU so they're real tensors
    # (not meta) before mark_sharding / .to(device). Each layer holds its
    # own compressed_kv / k_pe buffers, already initialized in Transformer
    # __init__ — but if the model was built on meta, those buffers landed
    # on meta too. Re-run early_initialization directly on CPU.
    mla0 = model.layers[0].attn
    cache_dtype = mla0.wkv_a.weight.dtype
    for cache_layer in getattr(model, "_cache_layers", []) or []:
        cache_layer.is_initialized = False
        cache_layer.early_initialization(
            batch_size=args.max_batch_size,
            kv_lora_rank=mla0.kv_lora_rank,
            pe_rank=mla0.qk_rope_head_dim,
            dtype=cache_dtype,
            device=torch.device("cpu"),
        )
    for layer in model.layers:
        attn = layer.attn
        attn.hadamard_matrix = hadamard
        if attn.indexer is not None:
            attn.indexer.k_cache = torch.zeros(
                args.max_batch_size,
                args.max_seq_len,
                attn.indexer.head_dim,
                dtype=torch.bfloat16,
            )
