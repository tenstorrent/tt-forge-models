# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading GGUF-quantized Lumina2 models.
"""

import torch
import diffusers.loaders.single_file_model as _sfm
import diffusers.loaders.single_file_utils as _sfu
from diffusers import GGUFQuantizationConfig, Lumina2Transformer2DModel
from huggingface_hub import hf_hub_download

# Lumina2 GGUF config for this checkpoint (inferred from tensor shapes)
_LUMINA2_HIDDEN_SIZE = 3840
_LUMINA2_NUM_LAYERS = 30
_LUMINA2_NUM_HEADS = 30
_LUMINA2_NUM_KV_HEADS = 30
_LUMINA2_CAP_FEAT_DIM = 2560
# FFN dim = 2/3 * 4 * hidden_size = 10240 (LLaMA-style SwiGLU multiplier)
_LUMINA2_FFN_DIM_MULTIPLIER = 2.0 / 3.0


def _patched_convert_lumina2_to_diffusers(checkpoint, **kwargs):
    """Patched version of convert_lumina2_to_diffusers that infers QKV split dims
    from the tensor shape, supporting both standard GQA and full-attention variants."""
    converted_state_dict = {}

    checkpoint.pop("norm_final.weight", None)

    keys = list(checkpoint.keys())
    for k in keys:
        if "model.diffusion_model." in k:
            checkpoint[k.replace("model.diffusion_model.", "")] = checkpoint.pop(k)

    LUMINA_KEY_MAP = {
        "cap_embedder": "time_caption_embed.caption_embedder",
        "t_embedder.mlp.0": "time_caption_embed.timestep_embedder.linear_1",
        "t_embedder.mlp.2": "time_caption_embed.timestep_embedder.linear_2",
        "attention": "attn",
        ".out.": ".to_out.0.",
        "k_norm": "norm_k",
        "q_norm": "norm_q",
        "w1": "linear_1",
        "w2": "linear_2",
        "w3": "linear_3",
        "adaLN_modulation.1": "norm1.linear",
    }
    ATTENTION_NORM_MAP = {
        "attention_norm1": "norm1.norm",
        "attention_norm2": "norm2",
    }
    CONTEXT_REFINER_MAP = {
        "context_refiner.0.attention_norm1": "context_refiner.0.norm1",
        "context_refiner.0.attention_norm2": "context_refiner.0.norm2",
        "context_refiner.1.attention_norm1": "context_refiner.1.norm1",
        "context_refiner.1.attention_norm2": "context_refiner.1.norm2",
    }
    FINAL_LAYER_MAP = {
        "final_layer.adaLN_modulation.1": "norm_out.linear_1",
        "final_layer.linear": "norm_out.linear_2",
    }

    # Derive QKV dims from config if available, else fall back to defaults
    config = kwargs.get("config", {})
    hidden_size = config.get("hidden_size", 2304)
    num_attention_heads = config.get("num_attention_heads", 24)
    num_kv_heads = config.get("num_kv_heads", 8)
    head_dim = hidden_size // num_attention_heads
    _q_dim = num_attention_heads * head_dim
    _k_dim = num_kv_heads * head_dim
    _v_dim = num_kv_heads * head_dim

    def convert_lumina_attn_to_diffusers(tensor, diffusers_key):
        to_q, to_k, to_v = torch.split(tensor, [_q_dim, _k_dim, _v_dim], dim=0)

        return {
            diffusers_key.replace("qkv", "to_q"): to_q,
            diffusers_key.replace("qkv", "to_k"): to_k,
            diffusers_key.replace("qkv", "to_v"): to_v,
        }

    keys = list(checkpoint.keys())
    for key in keys:
        diffusers_key = key
        for k, v in CONTEXT_REFINER_MAP.items():
            diffusers_key = diffusers_key.replace(k, v)
        for k, v in FINAL_LAYER_MAP.items():
            diffusers_key = diffusers_key.replace(k, v)
        for k, v in ATTENTION_NORM_MAP.items():
            diffusers_key = diffusers_key.replace(k, v)
        for k, v in LUMINA_KEY_MAP.items():
            diffusers_key = diffusers_key.replace(k, v)

        if "qkv" in diffusers_key:
            converted_state_dict.update(
                convert_lumina_attn_to_diffusers(checkpoint.pop(key), diffusers_key)
            )
        else:
            converted_state_dict[diffusers_key] = checkpoint.pop(key)

    return converted_state_dict


# Patch the conversion function before any model loading
_sfu.convert_lumina2_to_diffusers = _patched_convert_lumina2_to_diffusers
_sfm.SINGLE_FILE_LOADABLE_CLASSES["Lumina2Transformer2DModel"][
    "checkpoint_mapping_fn"
] = _patched_convert_lumina2_to_diffusers


def load_lumina2_transformer(
    repo_id: str, gguf_filename: str
) -> Lumina2Transformer2DModel:
    """Load a Lumina2 transformer from a GGUF checkpoint.

    Args:
        repo_id: HuggingFace repository ID.
        gguf_filename: Filename of the GGUF checkpoint within the repo.

    Returns:
        Lumina2Transformer2DModel: Loaded transformer set to eval mode.
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)

    quantization_config = GGUFQuantizationConfig(compute_dtype=torch.float32)

    transformer = Lumina2Transformer2DModel.from_single_file(
        model_path,
        quantization_config=quantization_config,
        torch_dtype=torch.float32,
        hidden_size=_LUMINA2_HIDDEN_SIZE,
        num_layers=_LUMINA2_NUM_LAYERS,
        num_attention_heads=_LUMINA2_NUM_HEADS,
        num_kv_heads=_LUMINA2_NUM_KV_HEADS,
        cap_feat_dim=_LUMINA2_CAP_FEAT_DIM,
        ffn_dim_multiplier=_LUMINA2_FFN_DIM_MULTIPLIER,
    )

    transformer.to("cpu")
    transformer.eval()

    for param in transformer.parameters():
        param.requires_grad = False

    return transformer


def create_lumina2_inputs(
    batch_size: int = 1,
    in_channels: int = 16,
    latent_height: int = 64,
    latent_width: int = 64,
    seq_len: int = 64,
    cap_feat_dim: int = _LUMINA2_CAP_FEAT_DIM,
):
    """Create synthetic inputs for Lumina2 transformer forward pass.

    Returns:
        tuple: (hidden_states, timestep, encoder_hidden_states, encoder_attention_mask)
    """
    hidden_states = torch.randn(batch_size, in_channels, latent_height, latent_width)
    timestep = torch.tensor([1.0] * batch_size)
    encoder_hidden_states = torch.randn(batch_size, seq_len, cap_feat_dim)
    encoder_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    return hidden_states, timestep, encoder_hidden_states, encoder_attention_mask
