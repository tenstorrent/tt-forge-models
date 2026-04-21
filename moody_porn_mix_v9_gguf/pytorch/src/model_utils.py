# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading GGUF-quantized Lumina2 models.
"""

import builtins as _builtins

import diffusers.loaders.single_file_model as _sfm
import diffusers.models.normalization as _norm_mod
import diffusers.models.transformers.transformer_lumina2 as _t2_mod
import torch
from diffusers import GGUFQuantizationConfig, Lumina2Transformer2DModel
from huggingface_hub import hf_hub_download

# Lumina2 architecture constants derived from GGUF tensor shapes:
#   x_embedder.weight: [64, 3840] => in_channels=16 (64 = 16*2*2 patch), hidden=3840
#   cap_embedder.0.weight: [2560] => text encoder (Qwen3-4B) hidden size
#   attention.qkv.weight: [3840, 11520] => q_dim=k_dim=v_dim=3840 (full attention, 30 heads)
#   final_layer.adaLN_modulation.1.weight: BF16 [3840, 512] => cond_dim=256 (not 1024)
#   k_norm/q_norm.weight: [128] => head_dim=128 (=3840/30), sum(axes_dim_rope) must be 128
IN_CHANNELS = 16
CAP_FEAT_DIM = 2560
HIDDEN_SIZE = 3840
NUM_LAYERS = 30
NUM_ATTENTION_HEADS = 30
NUM_KV_HEADS = 30
COND_DIM = 256  # conditioning dim; diffusers hardcodes min(hidden_size, 1024)=1024 but this model uses 256
# head_dim=128 so RoPE axes must sum to 128; keep time-axis at 32 (same as default) and
# scale the two spatial axes from 32 to 48 each: 32+48+48=128
AXES_DIM_ROPE = (32, 48, 48)


def _patch_cond_dim():
    """Patch diffusers Lumina2 classes to build with COND_DIM=256 conditioning.

    Three locations hardcode the conditioning dim incorrectly for this model:
    1. LuminaRMSNormZero.linear: uses min(embedding_dim, 1024)=1024, should be 256.
    2. norm_out.linear_1: uses min(hidden_size, 1024)=1024, should be 256.
    3. timestep_embedder: uses time_embed_dim=min(hidden_size,1024)=1024 (correct),
       but needs out_dim=COND_DIM=256 to output a 256-dim conditioning vector.

    Fixes 1 and 2 work by replacing min() in the relevant modules. Fix 3 directly
    overrides Lumina2CombinedTimestepCaptionEmbedding.__init__ because patching min()
    would incorrectly shrink time_embed_dim (the intermediate dim) from 1024 to 256.
    """
    import torch.nn as _nn

    def _min_capped(*args, **kwargs):
        result = _builtins.min(*args, **kwargs)
        return COND_DIM if result == 1024 else result

    _norm_mod.min = _min_capped
    _t2_mod.min = _min_capped

    _orig_tce_init = _t2_mod.Lumina2CombinedTimestepCaptionEmbedding.__init__

    def _patched_tce_init(
        self,
        hidden_size=4096,
        cap_feat_dim=2048,
        frequency_embedding_size=256,
        norm_eps=1e-5,
    ):
        _nn.Module.__init__(self)
        self.time_proj = _t2_mod.Timesteps(
            num_channels=frequency_embedding_size,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        # time_embed_dim stays at min(hidden_size, 1024)=1024 (GGUF linear_1 output is 1024);
        # out_dim=COND_DIM shrinks the final output to 256 (GGUF linear_2 output is 256).
        self.timestep_embedder = _t2_mod.TimestepEmbedding(
            in_channels=frequency_embedding_size,
            time_embed_dim=_builtins.min(hidden_size, 1024),
            out_dim=COND_DIM,
        )
        self.caption_embedder = _nn.Sequential(
            _t2_mod.RMSNorm(cap_feat_dim, eps=norm_eps),
            _nn.Linear(cap_feat_dim, hidden_size, bias=True),
        )

    _t2_mod.Lumina2CombinedTimestepCaptionEmbedding.__init__ = _patched_tce_init
    return _orig_tce_init


def _unpatch_cond_dim(orig_tce_init):
    """Remove the patches injected by _patch_cond_dim."""
    _t2_mod.Lumina2CombinedTimestepCaptionEmbedding.__init__ = orig_tce_init
    for mod in (_norm_mod, _t2_mod):
        mod.__dict__.pop("min", None)


def _patched_convert_lumina2_to_diffusers(checkpoint, **kwargs):
    """Patched version of convert_lumina2_to_diffusers with correct QKV split dims.

    The original function hardcodes dims for the base Lumina-Image-2.0 model
    (q_dim=2304, k_dim=v_dim=768). This model has hidden_size=3840 with full
    attention (q_dim=k_dim=v_dim=3840), so we need to override the split.
    """
    import torch as _torch

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
        # GGUF stores transformer-block adaLN as index 0 (the Linear directly); the
        # original non-GGUF Lumina2 uses adaLN_modulation = Sequential([SiLU, Linear])
        # so index 1 is the Linear. Both map to norm1.linear in diffusers.
        "adaLN_modulation.0": "norm1.linear",
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

    def convert_attn(tensor, diffusers_key):
        to_q, to_k, to_v = _torch.split(
            tensor, [HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE], dim=0
        )
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
                convert_attn(checkpoint.pop(key), diffusers_key)
            )
        else:
            converted_state_dict[diffusers_key] = checkpoint.pop(key)

    return converted_state_dict


def load_lumina2_transformer(
    repo_id: str, gguf_filename: str, compute_dtype=torch.bfloat16
):
    """Load a Lumina2 transformer from a GGUF checkpoint.

    Args:
        repo_id: HuggingFace repository ID.
        gguf_filename: Filename of the GGUF checkpoint within the repo.
        compute_dtype: Dtype for computation.

    Returns:
        Lumina2Transformer2DModel in eval mode.
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)
    quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

    # Patch SINGLE_FILE_LOADABLE_CLASSES so from_single_file uses our corrected QKV split
    # dims (3840+3840+3840 full attention) instead of the default (2304+768+768 GQA).
    # Also patch min() in diffusers norm/transformer modules so conditioning_dim uses
    # COND_DIM=256 instead of the hardcoded min(hidden_size, 1024)=1024.
    _lumina_entry = _sfm.SINGLE_FILE_LOADABLE_CLASSES["Lumina2Transformer2DModel"]
    _orig_fn = _lumina_entry["checkpoint_mapping_fn"]
    _lumina_entry["checkpoint_mapping_fn"] = _patched_convert_lumina2_to_diffusers
    _orig_tce_init = _patch_cond_dim()
    try:
        transformer = Lumina2Transformer2DModel.from_single_file(
            model_path,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_attention_heads=NUM_ATTENTION_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            cap_feat_dim=CAP_FEAT_DIM,
            ffn_dim_multiplier=2 / 3,
            axes_dim_rope=AXES_DIM_ROPE,
        )
    finally:
        _lumina_entry["checkpoint_mapping_fn"] = _orig_fn
        _unpatch_cond_dim(_orig_tce_init)

    transformer.eval()
    for param in transformer.parameters():
        param.requires_grad = False

    return transformer


def make_lumina2_inputs(
    dtype=torch.bfloat16,
    batch_size=1,
    height=128,
    width=128,
    max_sequence_length=128,
):
    """Build synthetic inputs for a Lumina2 transformer.

    Args:
        dtype: Tensor dtype.
        batch_size: Number of samples.
        height: Latent height (image_height / 8).
        width: Latent width (image_width / 8).
        max_sequence_length: Text token sequence length.

    Returns:
        dict of input tensors.
    """
    hidden_states = torch.randn(batch_size, IN_CHANNELS, height, width, dtype=dtype)
    timestep = torch.tensor([1.0 / 1000], dtype=dtype).expand(batch_size)
    encoder_hidden_states = torch.randn(
        batch_size, max_sequence_length, CAP_FEAT_DIM, dtype=dtype
    )
    encoder_attention_mask = torch.ones(
        batch_size, max_sequence_length, dtype=torch.bool
    )

    return {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
    }
