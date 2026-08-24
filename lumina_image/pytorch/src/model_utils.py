# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders and wrappers for Lumina-Image-2.0.

Model: Alpha-VLLM/Lumina-Image-2.0
Pipeline: diffusers.Lumina2Pipeline
Components:
  - text_encoder: Gemma2Model
  - transformer:  Lumina2Transformer2DModel
  - vae:          AutoencoderKL
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

LUMINA_REPO_ID = "Alpha-VLLM/Lumina-Image-2.0"
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Inference shape constants
# ---------------------------------------------------------------------------

HEIGHT = 1024
WIDTH = 1024

# AutoencoderKL spatial scaling factor (vae downsamples by 8)
VAE_SCALE_FACTOR = 8
LATENT_H = HEIGHT // VAE_SCALE_FACTOR  # 128
LATENT_W = WIDTH // VAE_SCALE_FACTOR  # 128

# Lumina2 transformer config: in_channels=16, patch_size=2, hidden_size=2304
NUM_CHANNELS_LATENTS = 16
PATCH_SIZE = 2
TEXT_EMBED_DIM = 2304

# Text encoder (Gemma-2): vocab=256000, hidden=2304
GEMMA_VOCAB_SIZE = 256000
MAX_SEQ_LEN = 256  # default Lumina2Pipeline max_sequence_length

# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_text_encoder(pretrained_model_name: str, dtype: torch.dtype):
    """Load the Gemma2Model text encoder from the text_encoder subfolder."""
    from transformers import AutoModel

    return AutoModel.from_pretrained(
        pretrained_model_name,
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def _lumina_apply_rotary_emb_real(x, freqs_cis, **_upstream_kwargs):
    """Real-valued drop-in for diffusers' ``apply_rotary_emb``.

    Implements the complex path only -- the one both ``transformer_lumina2`` call
    sites use. Since the swap is installed on that module's own binding (see
    ``_patch_lumina_rope_real``), the real-valued callers elsewhere in diffusers
    -- Flux, HunyuanDiT, CogVideoX -- keep the upstream function and never reach
    this one, so there is no second path to select and no ``use_real`` to branch
    on. Upstream's remaining keywords (``use_real``, ``use_real_unbind_dim``,
    ``sequence_dim``) are swallowed so the swap stays call-compatible: the call
    sites pass ``use_real=False`` explicitly, which would otherwise be a
    ``TypeError``.

    diffusers' Lumina branch does::

        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_out = torch.view_as_real(x_rotated * freqs_cis.unsqueeze(2)).flatten(3)

    which lowers to ``stablehlo.complex`` (the real->complex build) plus a complex
    ``stablehlo.multiply``. tt-mlir has no TTIR legalization for either, so the
    SHLO->TTIR conversion leaves a live ``tensor<2xf32> -> complex<f32>``
    materialization and fails ("failed to legalize unresolved materialization
    from tensor<2xf32> to tensor<complex<f32>>").

    Expand the complex product ``(a+bi)(c+di) = (ac-bd) + (ad+bc)i`` into real
    elementwise ops, preserving the interleaved ``[real, imag]`` output layout
    that ``view_as_real`` produces. ``freqs_cis`` may still be complex (default)
    or already real-stacked ``[..., D//2, 2]`` (see
    ``_patch_lumina_freqs_cis_real``). Verified bit-exact against the original in
    tests/torch/models/lumina_image/verify_freqs.py.
    """
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)  # [B, S, H, D//2, 2]
    a, b = x_[..., 0], x_[..., 1]
    if torch.is_complex(freqs_cis):
        fr, fi = freqs_cis.real, freqs_cis.imag
    else:  # real-stacked [..., D//2, 2]
        fr, fi = freqs_cis[..., 0], freqs_cis[..., 1]
    c, d = fr.unsqueeze(2), fi.unsqueeze(2)  # broadcast over the head dim
    out = torch.stack([a * c - b * d, a * d + b * c], dim=-1).flatten(3)
    return out.type_as(x)


def _patch_lumina_rope_real():
    """Swap in the real-valued RoPE so no complex arithmetic reaches tt-mlir.

    ``apply_rotary_emb`` is imported by name into ``transformer_lumina2`` at
    module load, so the attention processor calls the module-local binding --
    patch it there. That is both necessary (patching
    ``diffusers.models.embeddings`` would be too late) and sufficient: it leaves
    every other model's RoPE on the upstream function.

    Applied unconditionally -- rebinding the same object, so repeat calls are a
    no-op and there is no first-call-only state to protect.
    """
    from diffusers.models.transformers import transformer_lumina2

    transformer_lumina2.apply_rotary_emb = _lumina_apply_rotary_emb_real


def _patch_lumina_freqs_cis_real(model):
    """Make the RoPE ``freqs_cis`` real-valued end to end, from the gather on.

    tt-mlir's ComplexDataTypeConversion pass decomposes complex ``Gather``,
    ``Concatenate``, ``Reshape``, ``Slice``, ``BroadcastInDim``, ``Real`` and
    ``Imag`` into a trailing ``2xf32``, but it does NOT handle complex ``Pad`` or
    ``Select``. The embedder builds ``cap_freqs_cis``/``img_freqs_cis`` with
    ``torch.zeros(..., dtype=complex)`` + slice-assignment, which lowers to
    complex ``pad`` + ``select`` -- those can't be decomposed, so SHLO->TTIR
    fails with an unresolved ``tensor<...xcomplex<f32>> -> tensor<...x2xf32>``
    materialization (this is the pipeline failure the transformer test dodged
    because its freqs_cis path is only gather/concat/reshape/slice).
    """
    import types

    emb = model.rope_embedder
    if getattr(emb, "_tt_real_freqs", False):
        return

    def _get_freqs_cis(self, ids):
        """Real-stacked ``[B, S, sum(d_i)//2, 2]`` gather, no complex tensor.

        Mirrors the upstream gather one-for-one: the index is still broadcast
        across the frequency axis of a per-batch-repeated table, only now that
        axis is ``d`` (interleaved real/imag) instead of ``d//2`` complex.
        """
        result = []
        for i in range(len(self.axes_dim)):
            # [axes_len, d//2, 2] -> [axes_len, d], interleaved [real, imag].
            freqs = self.freqs_cis[i].to(ids.device).flatten(1)
            index = ids[:, :, i : i + 1].repeat(1, 1, freqs.shape[-1]).to(torch.int64)
            gathered = torch.gather(
                freqs.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index
            )
            result.append(gathered.reshape(*gathered.shape[:-1], -1, 2))
        # Concatenate on the frequency axis, not the [real, imag] pair axis.
        return torch.cat(result, dim=-2)

    def forward(self, hidden_states, attention_mask):
        batch_size, channels, height, width = hidden_states.shape
        p = self.patch_size
        post_patch_height, post_patch_width = height // p, width // p
        image_seq_len = post_patch_height * post_patch_width
        device = hidden_states.device

        encoder_seq_len = attention_mask.shape[1]
        l_effective_cap_len = attention_mask.sum(dim=1).tolist()
        seq_lengths = [cap + image_seq_len for cap in l_effective_cap_len]
        max_seq_len = max(seq_lengths)

        position_ids = torch.zeros(
            batch_size, max_seq_len, 3, dtype=torch.int32, device=device
        )
        for i, (cap_seq_len, seq_len) in enumerate(
            zip(l_effective_cap_len, seq_lengths)
        ):
            position_ids[i, :cap_seq_len, 0] = torch.arange(
                cap_seq_len, dtype=torch.int32, device=device
            )
            position_ids[i, cap_seq_len:seq_len, 0] = cap_seq_len
            row_ids = (
                torch.arange(post_patch_height, dtype=torch.int32, device=device)
                .view(-1, 1)
                .repeat(1, post_patch_width)
                .flatten()
            )
            col_ids = (
                torch.arange(post_patch_width, dtype=torch.int32, device=device)
                .view(1, -1)
                .repeat(post_patch_height, 1)
                .flatten()
            )
            position_ids[i, cap_seq_len:seq_len, 1] = row_ids
            position_ids[i, cap_seq_len:seq_len, 2] = col_ids

        # Already real-stacked [B, max_seq_len, D//2, 2] -- the patched
        # _get_freqs_cis gathers from real tables, so the zeros/pad/select below
        # (which tt-mlir cannot decompose on a complex type) stay real.
        freqs_cis = self._get_freqs_cis(position_ids)

        cap_freqs_cis = torch.zeros(
            batch_size,
            encoder_seq_len,
            *freqs_cis.shape[2:],
            device=device,
            dtype=freqs_cis.dtype,
        )
        img_freqs_cis = torch.zeros(
            batch_size,
            image_seq_len,
            *freqs_cis.shape[2:],
            device=device,
            dtype=freqs_cis.dtype,
        )
        for i, (cap_seq_len, seq_len) in enumerate(
            zip(l_effective_cap_len, seq_lengths)
        ):
            cap_freqs_cis[i, :cap_seq_len] = freqs_cis[i, :cap_seq_len]
            img_freqs_cis[i, :image_seq_len] = freqs_cis[i, cap_seq_len:seq_len]

        hidden_states = (
            hidden_states.view(
                batch_size, channels, post_patch_height, p, post_patch_width, p
            )
            .permute(0, 2, 4, 3, 5, 1)
            .flatten(3)
            .flatten(1, 2)
        )

        return (
            hidden_states,
            cap_freqs_cis,
            img_freqs_cis,
            freqs_cis,
            l_effective_cap_len,
            seq_lengths,
        )

    emb._get_freqs_cis = types.MethodType(_get_freqs_cis, emb)
    emb.forward = types.MethodType(forward, emb)
    emb._tt_real_freqs = True


def load_transformer(pretrained_model_name: str, dtype: torch.dtype):
    """Load Lumina2Transformer2DModel from the transformer subfolder."""
    from diffusers import Lumina2Transformer2DModel

    model = Lumina2Transformer2DModel.from_pretrained(
        pretrained_model_name,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()

    model.rope_embedder.freqs_cis = [
        torch.view_as_real(f.to(torch.complex64)).contiguous()
        for f in model.rope_embedder.freqs_cis
    ]

    _patch_lumina_rope_real()
    _patch_lumina_freqs_cis_real(model)

    return model


def load_vae(pretrained_model_name: str, dtype: torch.dtype):
    """Load AutoencoderKL from the vae subfolder."""
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


# ---------------------------------------------------------------------------
# Synthetic input builders
# ---------------------------------------------------------------------------


def load_text_encoder_inputs(dtype: torch.dtype):
    """Inputs for Gemma2Model: [input_ids, attention_mask].

    Shapes match what Lumina2Pipeline.encode_prompt feeds into the text
    encoder when max_sequence_length=256: input_ids (1, 256) and
    attention_mask (1, 256), both int64.
    """
    input_ids = torch.randint(0, GEMMA_VOCAB_SIZE, (1, MAX_SEQ_LEN), dtype=torch.long)
    attention_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)
    return [input_ids, attention_mask]


def load_transformer_inputs(dtype: torch.dtype, caption_len: int = MAX_SEQ_LEN):
    """Inputs for Lumina2Transformer2DModel: [hidden_states, timestep,
    encoder_hidden_states, encoder_attention_mask].

    Shapes match a live forward through Lumina2Transformer2DModel:
      hidden_states          (1, 16, 128, 128)   — 1024x1024 image latent
      timestep               (1,)  == tensor([0.])
      encoder_hidden_states  (1, 256, 2304)      — Gemma-2 hidden_size
      encoder_attention_mask (1, 256)            — ``caption_len`` leading ones
                              then zeros, i.e. the padding mask the pipeline's
                              tokenizer produces for a prompt of that length.

    ``caption_len`` selects how much of the fixed 256-slot caption buffer is
    real, and it changes *which code paths run* -- not just magnitudes:

      - ``256`` (the default, and what the pipeline never sees): the mask is
        all-ones, so the bool ``attn_mask`` the context refiners hand to
        ``scaled_dot_product_attention`` contains no ``False`` at all and is a
        semantic no-op. The joint sequence is 256 + 4096 = 4352 and the
        slice-assign offset is 256 -- both exact multiples of the 32-row tile.
      - anything smaller: 256 - caption_len keys must actually be masked out in
        the refiners, and the joint sequence (caption_len + 4096) and the
        joint-assembly / unpatchify slice offsets become caption_len-dependent,
        and tile-unaligned whenever caption_len % 32 != 0.

    A real prompt lands in the tens (see PROMPT in ``pipeline.py``), so pick
    values that separate those two effects: an aligned length < 256 keeps the
    slices tile-aligned while still exercising a masked softmax, whereas an
    unaligned length exercises both.
    """
    if not 0 < caption_len <= MAX_SEQ_LEN:
        raise ValueError(
            f"caption_len must be in (0, {MAX_SEQ_LEN}], got {caption_len}"
        )
    encoder_attention_mask = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)
    encoder_attention_mask[:, :caption_len] = 1
    encoder_hidden_states = torch.randn(1, MAX_SEQ_LEN, TEXT_EMBED_DIM, dtype=dtype)

    hidden_states = torch.randn(
        1, NUM_CHANNELS_LATENTS, LATENT_H, LATENT_W, dtype=dtype
    )
    timestep = torch.zeros(1, dtype=torch.float32)
    return [hidden_states, timestep, encoder_hidden_states, encoder_attention_mask]


def load_vae_inputs(dtype: torch.dtype):
    """Inputs for VAEDecoderWrapper: [z]."""
    z = torch.randn(1, NUM_CHANNELS_LATENTS, LATENT_H, LATENT_W, dtype=dtype)
    return [z]


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------


class Gemma2TextEncoderWrapper(torch.nn.Module):
    """Run Gemma2Model as a stateless text encoder returning a plain tensor.

    Pins use_cache=False so no KV cache is built. With a cache, Gemma-2's
    sliding-window layer slices the value states with
    full_value_states[:, :, -sliding_window + 1 :, :] (sliding_window=4096),
    producing slice index -4095 which exceeds the tt-mlir slice bound of
    [-256, 255] (tenstorrent/tt-xla#4900). A single encode pass needs no
    cache, so disabling it removes the offending slice entirely. Also pins
    return_dict=False so graph capture sees a pure tensor (last_hidden_state).
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )[0]


class Lumina2TransformerWrapper(torch.nn.Module):
    """Simplify Lumina2Transformer2DModel forward to return a plain tensor.

    The raw forward returns a Transformer2DModelOutput dataclass when
    return_dict=True. This wrapper pins return_dict=False and unwraps the
    single-element tuple so downstream graph capture sees a pure tensor.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask
    ):
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]


class VAEDecoderWrapper(torch.nn.Module):
    """Expose only the decoder half of AutoencoderKL as (z) -> tensor.

    The default vae(z) runs encode+decode and returns a ModelOutput object.
    This wrapper calls decode directly and unwraps the output to a plain tensor.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


# ---------------------------------------------------------------------------
# SPMD shard specifications
# ---------------------------------------------------------------------------

# (batch, model) mesh shapes by device count
MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


def shard_text_encoder_specs(encoder) -> dict:
    """Shard specs for Gemma2Model.

    Mesh axes: ("batch", "model")
    Column-parallel (q, k, v, gate, up): ("model", "batch")
    Row-parallel   (o, down):            ("batch", "model")
    """
    specs = {encoder.embed_tokens.weight: (None, "batch")}

    for layer in encoder.layers:
        sa = layer.self_attn
        specs[sa.q_proj.weight] = ("model", "batch")
        specs[sa.k_proj.weight] = ("model", "batch")
        specs[sa.v_proj.weight] = ("model", "batch")
        specs[sa.o_proj.weight] = ("batch", "model")

        mlp = layer.mlp
        specs[mlp.gate_proj.weight] = ("model", "batch")
        specs[mlp.up_proj.weight] = ("model", "batch")
        specs[mlp.down_proj.weight] = ("batch", "model")

        specs[layer.input_layernorm.weight] = ("batch",)
        specs[layer.post_attention_layernorm.weight] = ("batch",)
        specs[layer.pre_feedforward_layernorm.weight] = ("batch",)
        specs[layer.post_feedforward_layernorm.weight] = ("batch",)

    specs[encoder.norm.weight] = ("batch",)
    return specs


def _shard_lumina_block(block, specs: dict, has_adaln: bool) -> None:
    """Add shard specs for a single Lumina2TransformerBlock in-place.

    Single-axis Megatron tensor parallelism on the "model" axis only. Each
    block is two column->row pairs (attention Q/K/V -> O, FFN gate/up -> down),
    so it emits exactly two all-reduces and never a reshard. The block's input
    and output stay replicated (post-all-reduce), which lets the next block's
    column-parallel ops consume them directly.
    """
    attn = block.attn
    # Column-parallel: shard output (head) dim. GQA-safe (24 q-heads, 8 kv-heads
    # both divide by the model-axis size); to_q/k/v take a replicated input.
    specs[attn.to_q.weight] = ("model", None)
    specs[attn.to_k.weight] = ("model", None)
    specs[attn.to_v.weight] = ("model", None)
    # Row-parallel: shard contraction dim -> one all-reduce, replicated output.
    specs[attn.to_out[0].weight] = (None, "model")

    ff = block.feed_forward
    # SwiGLU column->row pair: gate/up column-parallel, down row-parallel.
    specs[ff.linear_1.weight] = ("model", None)
    specs[ff.linear_3.weight] = ("model", None)
    specs[ff.linear_2.weight] = (None, "model")


def shard_transformer_specs(transformer) -> dict:
    """Shard specs for Lumina2Transformer2DModel.

    Mesh axes: ("batch", "model"). Strategy: single-axis Megatron tensor
    parallelism on "model" for the 30 transformer blocks; everything else
    replicated.
      Column-parallel (Q, K, V, FFN up/gate): ("model", None)
      Row-parallel    (O, FFN down):          (None, "model")

    Embedders and the output projection (x_embedder, time_caption_embed,
    norm_out) are replicated: they sit at graph boundaries, are tiny relative to
    the blocks, and replicating them keeps block inputs/outputs replicated so no
    boundary reshard or final all-gather is needed. CCL budget: 60 all-reduces
    (2 per block x 30), 0 collective_permute, 0 all-gather. See
    sharding_analysis.md.
    """
    specs: dict = {}

    for block in transformer.noise_refiner:
        _shard_lumina_block(block, specs, has_adaln=True)
    for block in transformer.context_refiner:
        _shard_lumina_block(block, specs, has_adaln=False)
    for block in transformer.layers:
        _shard_lumina_block(block, specs, has_adaln=True)

    return specs


def _shard_resnet_block_megatron(block, specs: dict) -> None:
    """Megatron-style channel sharding within a diffusers ResnetBlock2D.

    Pairs the block's two convs so that only one all_reduce is needed per block:
      conv1 → column-parallel: weight (out, in, kH, kW) sharded on dim 0
              → ("model", None, None, None); bias sharded → ("model",)
      conv2 → row-parallel:    weight (out, out, kH, kW) sharded on dim 1
              → (None, "model", None, None); bias replicated(added once
              after all_reduce).
    conv_shortcut consumes the original replicated input and stays replicated
    so the residual add x + h matches the full output of conv2's all_reduce.
    """
    specs[block.conv1.weight] = ("model", None, None, None)
    if block.conv1.bias is not None:
        specs[block.conv1.bias] = ("model",)

    specs[block.conv2.weight] = (None, "model", None, None)


def shard_vae_specs(vae) -> dict:
    """Shard specs for AutoencoderKL (decoder path used by VAEDecoderWrapper).

    Strategy: Megatron-style channel sharding along a single mesh axis —
    "model" — only. Sharding along both mesh axes would break on multi-axis
    meshes (e.g. Galaxy (8, 4)) because Conv2D/Conv3D in tt-mlir can currently
    only be partitioned on the channel dim. Within each ResnetBlock2D the
    two convs alternate column-parallel → row-parallel:

      ResnetBlock2D.conv1   → column-parallel  ("model", None, None, None)
                              bias              → ("model",)
      ResnetBlock2D.conv2   → row-parallel     (None, "model", None, None)
                              bias (b2)         → replicated (omitted)
      conv_shortcut, Up/Downsample.conv, conv_in, conv_out  → replicated
      GroupNorm, conv_norm_out, attention.group_norm        → replicated

    The mid-block Attention follows the standard Megatron MLP pattern, also
    single-axis on "model":
      to_q / to_k / to_v   → column-parallel  ("model", None), bias ("model",)
      to_out[0]            → row-parallel     (None, "model"), bias replicated

    conv_in (16 → 512) and conv_out (128 → 3) are replicated so each resnet
    pair starts from a full input; conv_out's 3 output channels also do not
    divide cleanly across the model axis.

    GroupNorm affine weights are not sharded: group statistics need the full
    channel dim, so sharding the norm's channel-aligned weight would
    desynchronize the local-stat math. The runtime all_gathers the channel-
    sharded activation before each GroupNorm, runs the norm on a replicated
    tensor, then the next row-parallel conv computes partial sums + all_reduce
    to bring the residual output back to replicated for the shortcut add.
    """
    specs: dict = {}
    decoder = vae.decoder

    # conv_in: replicated (feeds full input into first column-parallel conv)
    # → omitted from specs

    for up_block in decoder.up_blocks:
        for resnet in up_block.resnets:
            _shard_resnet_block_megatron(resnet, specs)
        # upsampler convs: standalone, replicated → omitted

    mid = decoder.mid_block
    for resnet in mid.resnets:
        _shard_resnet_block_megatron(resnet, specs)
    for attn in mid.attentions:
        specs[attn.to_q.weight] = ("model", None)
        specs[attn.to_q.bias] = ("model",)
        specs[attn.to_k.weight] = ("model", None)
        specs[attn.to_k.bias] = ("model",)
        specs[attn.to_v.weight] = ("model", None)
        specs[attn.to_v.bias] = ("model",)
        specs[attn.to_out[0].weight] = (None, "model")
        # to_out[0].bias: replicated (added once after all_reduce)

    # conv_out: replicated → omitted from specs

    return specs
