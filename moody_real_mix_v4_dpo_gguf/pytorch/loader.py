# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Moody Real Mix v4 DPO GGUF model loader implementation for text-to-image generation.

Loads the GGUF-quantized Lumina2Transformer2DModel from
Gthalmie1/moody-real-mix-v4-dpo-gguf, a DPO-tuned Lumina-Image-2.0 checkpoint.
"""

from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.nn as nn
from diffusers import GGUFQuantizationConfig, Lumina2Transformer2DModel
from diffusers.quantizers.gguf.utils import _dequantize_gguf_and_restore_linear
from huggingface_hub import hf_hub_download

# GGUFParameter.as_tensor() calls _make_subclass which re-dispatches through
# __torch_function__ on the subclass, causing infinite recursion under Dynamo.
# DisableTorchFunctionSubclass breaks the cycle.
from diffusers.quantizers.gguf.utils import GGUFParameter as _GGUFParameter


def _patched_as_tensor(self):
    with torch.DisableTorchFunctionSubclass():
        return torch.Tensor._make_subclass(torch.Tensor, self, self.requires_grad)


_GGUFParameter.as_tensor = _patched_as_tensor


# TT hardware does not support complex types or float64. The Lumina2 RoPE embedder
# (_precompute_freqs_cis) defaults to complex<f64> on non-MPS systems. Patch it to
# use real-arithmetic (cos, sin) in float32 instead, and patch apply_rotary_emb to
# do real-domain rotation when given a (cos, sin) tuple with use_real=False.

def _patched_precompute_freqs_cis(self, axes_dim, axes_lens, theta):
    from diffusers.models.embeddings import get_1d_rotary_pos_embed
    freqs_cis = []
    for d, e in zip(axes_dim, axes_lens):
        emb = get_1d_rotary_pos_embed(d, e, theta=theta, freqs_dtype=torch.float32)
        # emb is complex64; extract (cos, sin) as float32 real tensors
        freqs_cis.append((emb.real.clone(), emb.imag.clone()))
    return freqs_cis


def _patched_get_freqs_cis(self, ids: torch.Tensor):
    # When freqs_cis entries are (cos, sin) float32 tuples (from _patched_precompute_freqs_cis),
    # gather each component separately and return a (cos, sin) tuple to avoid complex<f64> ops.
    if isinstance(self.freqs_cis[0], tuple):
        result_cos, result_sin = [], []
        for i, (cos, sin) in enumerate(self.freqs_cis):
            cos = cos.to(ids.device)
            sin = sin.to(ids.device)
            index = ids[:, :, i : i + 1].repeat(1, 1, cos.shape[-1]).to(torch.int64)
            result_cos.append(torch.gather(cos.unsqueeze(0).expand(index.shape[0], -1, -1), 1, index))
            result_sin.append(torch.gather(sin.unsqueeze(0).expand(index.shape[0], -1, -1), 1, index))
        return torch.cat(result_cos, dim=-1), torch.cat(result_sin, dim=-1)
    # Fallback: original complex path
    device = ids.device
    result = []
    for i in range(len(self.axes_dim)):
        freqs = self.freqs_cis[i].to(device)
        index = ids[:, :, i : i + 1].repeat(1, 1, freqs.shape[-1]).to(torch.int64)
        result.append(torch.gather(freqs.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
    return torch.cat(result, dim=-1).to(device)


def _patched_rope_forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    """Replacement for Lumina2RotaryPosEmbed.forward using fixed shapes, no .tolist().

    The original uses attention_mask.sum(dim=1).tolist() to get per-sample caption
    lengths, which requires a device-to-host transfer. Under the TT compiler this
    triggers partition_fx_graph_for_cpu_fallback which segfaults on the resulting
    graph structure. Replace with pure tensor ops using fixed encoder_seq_len from
    attention_mask.shape[1].
    """
    batch_size, channels, height, width = hidden_states.shape
    p = self.patch_size
    post_patch_height, post_patch_width = height // p, width // p
    image_seq_len = post_patch_height * post_patch_width
    device = hidden_states.device
    encoder_seq_len = attention_mask.shape[1]

    # Build position IDs using tensor ops only (no .cpu()/.tolist()).
    # Text tokens: axis0 = [0..encoder_seq_len-1], axes1,2 = 0
    text_axis0 = torch.arange(encoder_seq_len, dtype=torch.int32, device=device)
    zeros_enc = torch.zeros(encoder_seq_len, dtype=torch.int32, device=device)
    text_ids = torch.stack([text_axis0, zeros_enc, zeros_enc], dim=-1)  # [S_cap, 3]

    # Image tokens: axis0 = encoder_seq_len (fixed cap offset), axes1,2 = row,col
    row_ids = (
        torch.arange(post_patch_height, dtype=torch.int32, device=device)
        .view(-1, 1).expand(-1, post_patch_width).flatten()
    )
    col_ids = (
        torch.arange(post_patch_width, dtype=torch.int32, device=device)
        .view(1, -1).expand(post_patch_height, -1).flatten()
    )
    cap_offset = torch.full((image_seq_len,), encoder_seq_len, dtype=torch.int32, device=device)
    image_ids = torch.stack([cap_offset, row_ids, col_ids], dim=-1)  # [S_img, 3]

    ids = torch.cat([text_ids, image_ids], dim=0)  # [total_seq_len, 3]
    position_ids = ids.unsqueeze(0).expand(batch_size, -1, -1)  # [B, total_seq_len, 3]

    freqs_cis = self._get_freqs_cis(position_ids)

    if isinstance(freqs_cis, tuple):
        cos_all, sin_all = freqs_cis  # [B, total_seq_len, D_rope]
        context_rotary_emb = (cos_all[:, :encoder_seq_len], sin_all[:, :encoder_seq_len])
        noise_rotary_emb = (cos_all[:, encoder_seq_len:], sin_all[:, encoder_seq_len:])
    else:
        context_rotary_emb = freqs_cis[:, :encoder_seq_len]
        noise_rotary_emb = freqs_cis[:, encoder_seq_len:]

    hidden_states = (
        hidden_states.view(batch_size, channels, post_patch_height, p, post_patch_width, p)
        .permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
    )

    # Return encoder_seq_lengths and seq_lengths as Python lists (used by patched transformer
    # forward only as passthrough; not used for dynamic graph operations).
    l_effective_cap_len = [encoder_seq_len] * batch_size
    seq_lengths = [encoder_seq_len + image_seq_len] * batch_size
    return hidden_states, context_rotary_emb, noise_rotary_emb, freqs_cis, l_effective_cap_len, seq_lengths


import diffusers.models.embeddings as _emb
_orig_apply_rotary_emb = _emb.apply_rotary_emb


def _patched_apply_rotary_emb(x, freqs_cis, use_real=True, **kwargs):
    # When use_real=False but freqs_cis is a (cos, sin) float32 tuple (not complex),
    # use real-domain rotation to avoid complex<f64> tensors unsupported on TT.
    is_real_cos_sin = (
        not use_real
        and isinstance(freqs_cis, tuple)
        and len(freqs_cis) == 2
        and isinstance(freqs_cis[0], torch.Tensor)
        and not freqs_cis[0].is_complex()
    )
    if is_real_cos_sin:
        cos, sin = freqs_cis  # (B, S, D_rope) float32
        cos = cos.unsqueeze(2)  # (B, S, 1, D_rope)
        sin = sin.unsqueeze(2)
        # x: [B, S, H, head_dim] — sequence dim=1, head dim=2
        x_r, x_i = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
        out = torch.stack([x_r * cos - x_i * sin, x_r * sin + x_i * cos], dim=-1).flatten(3)
        return out.to(x.dtype)
    return _orig_apply_rotary_emb(x, freqs_cis, use_real=use_real, **kwargs)


_emb.apply_rotary_emb = _patched_apply_rotary_emb
import diffusers.models.transformers.transformer_lumina2 as _tl2_module
_tl2_module.apply_rotary_emb = _patched_apply_rotary_emb

# Apply RoPE method patches globally so they persist after model loading.
_tl2_module.Lumina2RotaryPosEmbed._precompute_freqs_cis = _patched_precompute_freqs_cis
_tl2_module.Lumina2RotaryPosEmbed._get_freqs_cis = _patched_get_freqs_cis
_tl2_module.Lumina2RotaryPosEmbed.forward = _patched_rope_forward


def _patched_transformer_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    attention_kwargs: Optional[dict] = None,
    return_dict: bool = True,
):
    """Replace for-loop-over-seq-lengths with tensor ops to avoid partition segfault.

    The original forward iterates per-batch over seq_lengths (Python ints derived
    from attention_mask.tolist()) to build joint_hidden_states and to unpatchify.
    Under the TT compiler these patterns trigger partition_fx_graph_for_cpu_fallback
    which segfaults. Replace with torch.cat (joint concat) and tensor slices.
    """
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    batch_size, _, height, width = hidden_states.shape

    temb, encoder_hidden_states = self.time_caption_embed(
        hidden_states, timestep, encoder_hidden_states
    )

    (
        hidden_states,
        context_rotary_emb,
        noise_rotary_emb,
        rotary_emb,
        _,  # encoder_seq_lengths — not used here
        _,  # seq_lengths — not used here
    ) = self.rope_embedder(hidden_states, encoder_attention_mask)

    hidden_states = self.x_embedder(hidden_states)

    for layer in self.context_refiner:
        encoder_hidden_states = layer(
            encoder_hidden_states, encoder_attention_mask, context_rotary_emb
        )

    for layer in self.noise_refiner:
        hidden_states = layer(hidden_states, None, noise_rotary_emb, temb)

    # Concatenate text and image token sequences without dynamic per-batch loops.
    # encoder_hidden_states: [B, encoder_seq_len, D]
    # hidden_states:         [B, image_seq_len, D]
    encoder_seq_len = encoder_attention_mask.shape[1]
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for layer in self.layers:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                layer, hidden_states, None, rotary_emb, temb
            )
        else:
            hidden_states = layer(hidden_states, None, rotary_emb, temb)

    hidden_states = self.norm_out(hidden_states, temb)

    # Unpatchify without for-loop: extract image tokens and reshape.
    p = self.config.patch_size
    image_seq_len = (height // p) * (width // p)

    # hidden_states: [B, encoder_seq_len + image_seq_len, D] → image tokens start at encoder_seq_len
    output = (
        hidden_states[:, encoder_seq_len : encoder_seq_len + image_seq_len]
        .view(batch_size, height // p, width // p, p, p, self.out_channels)
        .permute(0, 5, 1, 3, 2, 4)
        .flatten(4, 5)
        .flatten(2, 3)
    )

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


_tl2_module.Lumina2Transformer2DModel.forward = _patched_transformer_forward

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

REPO_ID = "Gthalmie1/moody-real-mix-v4-dpo-gguf"

# Architecture constants derived from GGUF tensor shapes:
# - hidden_size=3840, MHA 30 heads (head_dim=128, equal-thirds QKV)
# - cap_feat_dim=2560 (cap_embedder.0 norm dim in GGUF)
# - Timestep bottleneck: 256→1024→256 (output 256)
# - AdaLN conditioning dim: 256
IN_CHANNELS = 16
CAP_FEAT_DIM = 2560
_TIMESTEP_EMBED_DIM = 256


class ModelVariant(StrEnum):
    """Available Moody Real Mix v4 DPO GGUF model variants."""

    Q4_K_M = "Q4_K_M"


_GGUF_FILES = {
    ModelVariant.Q4_K_M: "moodyRealMix_zitV4DPO_q4_k_m.gguf",
}


@contextmanager
def _patch_lumina2_for_gguf():
    """Patch diffusers Lumina2 classes to load this GGUF variant.

    This GGUF uses a Lumina2 architectural variant that differs from the
    reference model supported by diffusers in four ways:

    1. QKV split: MHA equal-thirds (11520 → 3×3840) vs GQA (3840 → 2304+768+768)
    2. Timestep embedding: 256→1024→256 bottleneck (output 256) vs 256→1024 (output 1024)
    3. AdaLN conditioning dim: 256 (from bottleneck) vs min(hidden_size, 1024)=1024
    4. AdaLN sequential key: .0 (linear at index 0) vs .1 (SiLU at 0, linear at 1)
    """
    import diffusers.loaders.single_file_model as _sfm
    import diffusers.models.normalization as _norm
    import diffusers.models.transformers.transformer_lumina2 as _tl2
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
    from diffusers.models.normalization import RMSNorm

    # --- Patch 1: checkpoint converter ---
    _CLASS_ENTRY = _sfm.SINGLE_FILE_LOADABLE_CLASSES["Lumina2Transformer2DModel"]
    _orig_converter = _CLASS_ENTRY["checkpoint_mapping_fn"]

    def _patched_converter(checkpoint, **kwargs):
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
            "adaLN_modulation.0": "norm1.linear",
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

        def _split_qkv(tensor, diffusers_key):
            total_dim = tensor.shape[0]
            if total_dim == 2304 + 768 + 768:
                q_dim, k_dim, v_dim = 2304, 768, 768
            else:
                q_dim = k_dim = v_dim = total_dim // 3
            to_q, to_k, to_v = torch.split(tensor, [q_dim, k_dim, v_dim], dim=0)
            return {
                diffusers_key.replace("qkv", "to_q"): to_q,
                diffusers_key.replace("qkv", "to_k"): to_k,
                diffusers_key.replace("qkv", "to_v"): to_v,
            }

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
                    _split_qkv(checkpoint.pop(key), diffusers_key)
                )
            else:
                converted_state_dict[diffusers_key] = checkpoint.pop(key)
        return converted_state_dict

    _CLASS_ENTRY["checkpoint_mapping_fn"] = _patched_converter

    # --- Patch 2: LuminaRMSNormZero — use 256-dim conditioning input ---
    _orig_rms_init = _norm.LuminaRMSNormZero.__init__

    def _patched_rms_init(self, embedding_dim, norm_eps, norm_elementwise_affine):
        nn.Module.__init__(self)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(_TIMESTEP_EMBED_DIM, 4 * embedding_dim, bias=True)
        self.norm = RMSNorm(
            embedding_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine
        )

    _norm.LuminaRMSNormZero.__init__ = _patched_rms_init

    # --- Patch 3: Lumina2CombinedTimestepCaptionEmbedding — bottleneck timestep MLP ---
    _orig_tsembed_init = _tl2.Lumina2CombinedTimestepCaptionEmbedding.__init__

    def _patched_tsembed_init(self, hidden_size=4096, cap_feat_dim=2048, norm_eps=1e-6):
        nn.Module.__init__(self)
        freq_dim = 256
        self.time_proj = Timesteps(
            num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0.0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=freq_dim,
            time_embed_dim=4 * freq_dim,
            out_dim=freq_dim,
        )
        self.caption_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, hidden_size, bias=True),
        )

    _tl2.Lumina2CombinedTimestepCaptionEmbedding.__init__ = _patched_tsembed_init

    # --- Patch 4: LuminaLayerNormContinuous — use 256-dim conditioning for norm_out ---
    _orig_llnc_init = _norm.LuminaLayerNormContinuous.__init__

    def _patched_llnc_init(self, embedding_dim, conditioning_embedding_dim, **kwargs):
        if conditioning_embedding_dim > _TIMESTEP_EMBED_DIM:
            conditioning_embedding_dim = _TIMESTEP_EMBED_DIM
        _orig_llnc_init(self, embedding_dim, conditioning_embedding_dim, **kwargs)

    _norm.LuminaLayerNormContinuous.__init__ = _patched_llnc_init

    try:
        yield
    finally:
        _CLASS_ENTRY["checkpoint_mapping_fn"] = _orig_converter
        _norm.LuminaRMSNormZero.__init__ = _orig_rms_init
        _tl2.Lumina2CombinedTimestepCaptionEmbedding.__init__ = _orig_tsembed_init
        _norm.LuminaLayerNormContinuous.__init__ = _orig_llnc_init


class ModelLoader(ForgeModel):
    """Moody Real Mix v4 DPO GGUF model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Moody Real Mix v4 DPO GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        repo_id = self._variant_config.pretrained_model_name
        gguf_filename = _GGUF_FILES[self._variant]

        model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)
        with _patch_lumina2_for_gguf():
            self.transformer = Lumina2Transformer2DModel.from_single_file(
                model_path,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
                # Architecture overrides for this GGUF variant (hidden_size=3840, MHA)
                hidden_size=3840,
                num_layers=30,
                num_refiner_layers=2,
                num_attention_heads=30,
                num_kv_heads=30,
                multiple_of=256,
                ffn_dim_multiplier=2 / 3,
                cap_feat_dim=CAP_FEAT_DIM,
                axes_dim_rope=(32, 48, 48),
            )
        _dequantize_gguf_and_restore_linear(self.transformer)
        torch.nn.Module.to(self.transformer, compute_dtype)
        self.transformer.eval()
        return self.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transformer is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        height = 128
        width = 128
        hidden_states = torch.randn(batch_size, IN_CHANNELS, height, width, dtype=dtype)

        timestep = torch.tensor([1.0 / 1000], dtype=dtype).expand(batch_size)

        max_sequence_length = 128
        encoder_hidden_states = torch.randn(
            batch_size, max_sequence_length, CAP_FEAT_DIM, dtype=dtype
        )

        encoder_attention_mask = torch.ones(
            batch_size, max_sequence_length, dtype=torch.bool
        )

        inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }

        return inputs
