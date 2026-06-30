# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for HiDream-I1-Fast.

Model: HiDream-ai/HiDream-I1-Fast (Sparse-MoE MM-DiT, guidance- + step-distilled)
Components:
  - text_encoder    : CLIPTextModelWithProjection (CLIP ViT-L/14, ~0.123 B)
  - text_encoder_2  : CLIPTextModelWithProjection (OpenCLIP ViT-bigG/14, ~0.695 B)
  - text_encoder_3  : T5EncoderModel (T5 v1.1 XXL encoder, ~4.6 B)
  - text_encoder_4  : LlamaForCausalLM (Llama-3.1-8B-Instruct, ~8.0 B)
  - transformer     : HiDreamImageTransformer2DModel (Sparse-MoE MM-DiT, ~17 B)
  - vae             : AutoencoderKL (FLUX-derived 16-channel latent, ~0.084 B)
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

HIDREAM_REPO_ID = "HiDream-ai/HiDream-I1-Fast"
LLAMA_REPO_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DTYPE = torch.float32

# ---------------------------------------------------------------------------
# Inference shape constants (HiDream-I1-Fast @ 1024x1024)
# ---------------------------------------------------------------------------

HEIGHT = 1024
WIDTH = 1024
VAE_SCALE = 8

LATENT_H = HEIGHT // VAE_SCALE  # 128
LATENT_W = WIDTH // VAE_SCALE  # 128
LATENT_CHANNELS = 16

MAX_SEQ_LEN = 128  # pipeline truncates all text encoders to 128

# CLIP-L (text_encoder) and CLIP-G (text_encoder_2) — pooled output dims
CLIP_L_HIDDEN = 768
CLIP_G_HIDDEN = 1280
CLIP_VOCAB_SIZE = 49408

# T5 (text_encoder_3) — full token sequence
T5_HIDDEN = 4096
T5_VOCAB_SIZE = 32128

# Llama (text_encoder_4) — outputs stacked hidden states from all 32 layers
LLAMA_HIDDEN = 4096
LLAMA_NUM_LAYERS = 32
LLAMA_VOCAB_SIZE = 128256

# Transformer (DiT) — conditioning shapes
POOLED_TEXT_EMB_DIM = CLIP_L_HIDDEN + CLIP_G_HIDDEN  # 2048

# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_text_encoder(pretrained_model_name: str, dtype: torch.dtype):
    """Load CLIP-L (text_encoder subfolder) as CLIPTextModelWithProjection."""
    from transformers import CLIPTextModelWithProjection

    return CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_name,
        subfolder="text_encoder",
        torch_dtype=dtype,
    ).eval()


def load_text_encoder_2(pretrained_model_name: str, dtype: torch.dtype):
    """Load CLIP-G (text_encoder_2 subfolder) as CLIPTextModelWithProjection."""
    from transformers import CLIPTextModelWithProjection

    return CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_name,
        subfolder="text_encoder_2",
        torch_dtype=dtype,
    ).eval()


def load_text_encoder_3(pretrained_model_name: str, dtype: torch.dtype):
    """Load T5-XXL encoder (text_encoder_3 subfolder)."""
    from transformers import T5EncoderModel

    return T5EncoderModel.from_pretrained(
        pretrained_model_name,
        subfolder="text_encoder_3",
        torch_dtype=dtype,
    ).eval()


def load_text_encoder_4(pretrained_model_name: str, dtype: torch.dtype):
    """Load Llama-3.1-8B-Instruct as text_encoder_4.

    Loaded from the standalone Meta repo (HiDream's pipeline expects the user to
    supply this; the snapshot does not ship Llama weights).
    """
    from transformers import LlamaForCausalLM

    return LlamaForCausalLM.from_pretrained(
        pretrained_model_name,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=dtype,
    ).eval()


def _patch_hidream_moe_infer() -> None:
    """Drop the `.cpu().numpy()` round-trip in MOEFeedForwardSwiGLU.moe_infer —
    it forces a host sync + numpy array that torch.compile can't trace
    ("'ndarray' object has no attribute 'dim'"). Keeping bincount().cumsum(0) as
    on-device torch ops makes it traceable; everything else is unchanged.

    See https://github.com/tenstorrent/tt-xla/issues/5290
    """
    from diffusers.models.transformers.transformer_hidream_image import (
        MOEFeedForwardSwiGLU,
    )

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cumsum(0)
        token_idxs = idxs // self.num_activated_experts
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )
        return expert_cache

    MOEFeedForwardSwiGLU.moe_infer = moe_infer


def load_transformer(pretrained_model_name: str, dtype: torch.dtype):
    """Load HiDreamImageTransformer2DModel from the transformer subfolder."""
    from diffusers import HiDreamImageTransformer2DModel

    _patch_hidream_moe_infer()

    return HiDreamImageTransformer2DModel.from_pretrained(
        pretrained_model_name,
        subfolder="transformer",
        torch_dtype=dtype,
    ).eval()


def load_vae(pretrained_model_name: str, dtype: torch.dtype):
    """Load AutoencoderKL from the vae subfolder."""
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=dtype,
    ).eval()


# ---------------------------------------------------------------------------
# Wrapper modules — simplify component forward signatures to positional tensors.
# Each wrapper mimics what the HiDream pipeline does at the call site.
# ---------------------------------------------------------------------------


class CLIPPooledWrapper(torch.nn.Module):
    """Return the pooled CLIP embedding (text_embeds).

    Used for both CLIP-L (text_encoder, hidden=768) and CLIP-G (text_encoder_2,
    hidden=1280). Mirrors `_get_clip_prompt_embeds` which discards hidden states
    and keeps only `prompt_embeds[0]` (the projected pooled output of
    CLIPTextModelWithProjection).
    """

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        out = self.text_encoder(input_ids, output_hidden_states=True)
        return out[0]


class T5EncoderWrapper(torch.nn.Module):
    """Return T5 last hidden state.

    Mirrors `_get_t5_prompt_embeds`: `self.text_encoder_3(input_ids, attention_mask=...)[0]`.
    """

    def __init__(self, t5_encoder):
        super().__init__()
        self.t5_encoder = t5_encoder

    def forward(self, input_ids, attention_mask):
        return self.t5_encoder(input_ids, attention_mask=attention_mask)[0]


class LlamaStackedHiddenWrapper(torch.nn.Module):
    """Stack all Llama hidden states (skipping the embedding output).

    Mirrors `_get_llama3_prompt_embeds`: takes `outputs.hidden_states[1:]` and
    stacks along a new leading dim.
    """

    def __init__(self, llama):
        super().__init__()
        self.llama = llama

    def forward(self, input_ids, attention_mask):
        outputs = self.llama(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )
        stacked = torch.stack(outputs.hidden_states[1:], dim=0)
        return stacked


class HiDreamTransformerWrapper(torch.nn.Module):
    """Flatten the DiT forward signature to pure positional tensors.

    Inputs match the pipeline call site exactly. Returns the raw transformer
    output (noise_pred) before the pipeline's sign flip.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states,
        timesteps,
        encoder_hidden_states_t5,
        encoder_hidden_states_llama3,
        pooled_embeds,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            timesteps=timesteps,
            encoder_hidden_states_t5=encoder_hidden_states_t5,
            encoder_hidden_states_llama3=encoder_hidden_states_llama3,
            pooled_embeds=pooled_embeds,
            return_dict=False,
        )[0]


class VAEDecoderWrapper(torch.nn.Module):
    """Expose only the decoder half of AutoencoderKL as (z) -> tensor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


# ---------------------------------------------------------------------------
# SPMD shard specifications
# ---------------------------------------------------------------------------

# (batch, model) mesh shapes by device count.
# Matches krea_realtime's convention.
MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


def shard_llama_specs(llama) -> dict:
    """Shard specs for LlamaForCausalLM (text_encoder_4).

    Mesh axes: ("batch", "model")
    Column-parallel (Q, K, V, gate, up):  ("model", "batch")
    Row-parallel   (O, down):             ("batch", "model")
    LayerNorms (RMSNorm) sharded along batch.
    """
    specs = {}

    base = llama.model  # LlamaModel
    specs[base.embed_tokens.weight] = (None, "batch")

    for layer in base.layers:
        attn = layer.self_attn
        specs[attn.q_proj.weight] = ("model", "batch")
        specs[attn.k_proj.weight] = ("model", "batch")
        specs[attn.v_proj.weight] = ("model", "batch")
        specs[attn.o_proj.weight] = ("batch", "model")

        mlp = layer.mlp
        specs[mlp.gate_proj.weight] = ("model", "batch")
        specs[mlp.up_proj.weight] = ("model", "batch")
        specs[mlp.down_proj.weight] = ("batch", "model")

        specs[layer.input_layernorm.weight] = ("batch",)
        specs[layer.post_attention_layernorm.weight] = ("batch",)

    specs[base.norm.weight] = ("batch",)
    # lm_head is unused at inference (only hidden_states matter), but keep it
    # sharded so weight loading is balanced if it ends up materialized.
    if getattr(llama, "lm_head", None) is not None:
        specs[llama.lm_head.weight] = ("model", "batch")
    return specs


def shard_t5_encoder_specs(t5_encoder) -> dict:
    """Shard specs for T5EncoderModel (text_encoder_3, T5 v1.1 XXL).

    Mesh axes: ("batch", "model")
    Column-parallel (Q, K, V, wi_0, wi_1):  ("model", "batch")
    Row-parallel   (O, wo):                 ("batch", "model")
    T5LayerNorm weights sharded along batch.
    Token embedding sharded along d_model on the batch axis (matches Llama).
    relative_attention_bias (32 buckets x 64 heads) is tiny; left replicated.
    """
    specs = {}

    # Token embedding (tied: t5_encoder.shared and encoder.embed_tokens share weight).
    specs[t5_encoder.shared.weight] = (None, "batch")

    stack = t5_encoder.encoder  # T5Stack
    for block in stack.block:
        # layer[0]: T5LayerSelfAttention.
        attn_layer = block.layer[0]
        attn = attn_layer.SelfAttention
        specs[attn.q.weight] = ("model", "batch")
        specs[attn.k.weight] = ("model", "batch")
        specs[attn.v.weight] = ("model", "batch")
        specs[attn.o.weight] = ("batch", "model")
        specs[attn_layer.layer_norm.weight] = ("batch",)

        # layer[1]: T5LayerFF with T5DenseGatedActDense (v1.1).
        ff_layer = block.layer[1]
        dense = ff_layer.DenseReluDense
        specs[dense.wi_0.weight] = ("model", "batch")
        specs[dense.wi_1.weight] = ("model", "batch")
        specs[dense.wo.weight] = ("batch", "model")
        specs[ff_layer.layer_norm.weight] = ("batch",)

    specs[stack.final_layer_norm.weight] = ("batch",)
    return specs


def shard_hidream_transformer_specs(transformer) -> dict:
    """Shard specs for HiDreamImageTransformer2DModel.

    Mesh axes: ("batch", "model")
    Attention: column-parallel Q/K/V (("model", "batch")), row-parallel O.
    MoE experts: w1, w3 column-parallel; w2 row-parallel.
    Text-stream FF in double-stream blocks: same as MoE expert pattern.
    adaLN / RMSNorm / caption_projection sharded along batch.
    """
    specs = {}

    # Patch embedder: Linear(in_channels * patch_size^2 -> inner_dim).
    specs[transformer.x_embedder.proj.weight] = ("model", "batch")
    specs[transformer.x_embedder.proj.bias] = ("model",)

    # Timestep embedder (TimestepEmbedding has linear_1 + linear_2).
    t_emb = transformer.t_embedder.timestep_embedder
    specs[t_emb.linear_1.weight] = ("model", "batch")
    specs[t_emb.linear_1.bias] = ("model",)
    specs[t_emb.linear_2.weight] = ("batch", "model")
    specs[t_emb.linear_2.bias] = ("batch",)

    # Pooled embedder (CLIP-L+G pooled -> inner_dim).
    p_emb = transformer.p_embedder.pooled_embedder
    specs[p_emb.linear_1.weight] = ("model", "batch")
    specs[p_emb.linear_1.bias] = ("model",)
    specs[p_emb.linear_2.weight] = ("batch", "model")
    specs[p_emb.linear_2.bias] = ("batch",)

    # Caption projection (T5 + Llama-per-layer -> inner_dim).
    # TextProjection module has a `linear` submodule based on diffusers source.
    for proj in transformer.caption_projection:
        # Each TextProjection wraps a Linear named `linear`.
        if hasattr(proj, "linear"):
            specs[proj.linear.weight] = ("model", "batch")
            if proj.linear.bias is not None:
                specs[proj.linear.bias] = ("model",)

    def _shard_attention(attn, has_text_stream: bool):
        # Image stream Q/K/V/O.
        specs[attn.to_q.weight] = ("model", "batch")
        specs[attn.to_q.bias] = ("model",)
        specs[attn.to_k.weight] = ("model", "batch")
        specs[attn.to_k.bias] = ("model",)
        specs[attn.to_v.weight] = ("model", "batch")
        specs[attn.to_v.bias] = ("model",)
        specs[attn.to_out.weight] = ("batch", "model")
        specs[attn.to_out.bias] = ("batch",)
        if attn.q_rms_norm.weight is not None:
            specs[attn.q_rms_norm.weight] = ("batch",)
        if attn.k_rms_norm.weight is not None:
            specs[attn.k_rms_norm.weight] = ("batch",)
        if has_text_stream:
            specs[attn.to_q_t.weight] = ("model", "batch")
            specs[attn.to_q_t.bias] = ("model",)
            specs[attn.to_k_t.weight] = ("model", "batch")
            specs[attn.to_k_t.bias] = ("model",)
            specs[attn.to_v_t.weight] = ("model", "batch")
            specs[attn.to_v_t.bias] = ("model",)
            specs[attn.to_out_t.weight] = ("batch", "model")
            specs[attn.to_out_t.bias] = ("batch",)
            if attn.q_rms_norm_t.weight is not None:
                specs[attn.q_rms_norm_t.weight] = ("batch",)
            if attn.k_rms_norm_t.weight is not None:
                specs[attn.k_rms_norm_t.weight] = ("batch",)

    def _shard_swiglu(ffn):
        # HiDreamImageFeedForwardSwiGLU: w1 (gate, up), w3 (up), w2 (down). No bias.
        specs[ffn.w1.weight] = ("model", "batch")
        specs[ffn.w3.weight] = ("model", "batch")
        specs[ffn.w2.weight] = ("batch", "model")

    def _shard_moe(ff):
        # MOEFeedForwardSwiGLU: .experts (ModuleList of SwiGLU) and .gate (MoEGate).
        # The MoEGate weight is (n_experts=4, hidden) and drives top-k routing —
        # every device needs to see all expert scores to pick the same top-2
        # indices, so it must be replicated (not sharded). It's tiny (~10 KB)
        # so replication has negligible memory cost.
        for expert in ff.experts:
            _shard_swiglu(expert)

    # Double-stream blocks (16): image+text streams, image FF is MoE.
    for hidream_block in transformer.double_stream_blocks:
        blk = hidream_block.block
        # adaLN modulation: SiLU + Linear(dim, 12*dim).
        specs[blk.adaLN_modulation[1].weight] = ("model", "batch")
        specs[blk.adaLN_modulation[1].bias] = ("model",)
        _shard_attention(blk.attn1, has_text_stream=True)
        # Image FF (MoE) and text FF (dense SwiGLU).
        _shard_moe(blk.ff_i)
        _shard_swiglu(blk.ff_t)

    # Single-stream blocks (32): image only, FF is MoE.
    for hidream_block in transformer.single_stream_blocks:
        blk = hidream_block.block
        # adaLN modulation: SiLU + Linear(dim, 6*dim).
        specs[blk.adaLN_modulation[1].weight] = ("model", "batch")
        specs[blk.adaLN_modulation[1].bias] = ("model",)
        _shard_attention(blk.attn1, has_text_stream=False)
        _shard_moe(blk.ff_i)

    # Final layer: Linear(inner_dim -> patch_size^2 * out_channels) + adaLN(SiLU+Linear).
    specs[transformer.final_layer.linear.weight] = (None, "batch")
    specs[transformer.final_layer.linear.bias] = (None,)
    specs[transformer.final_layer.adaLN_modulation[1].weight] = ("model", "batch")
    specs[transformer.final_layer.adaLN_modulation[1].bias] = ("model",)

    return specs
