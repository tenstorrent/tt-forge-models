# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders, wrappers, shard specs and input builders for Infinity.

Infinity ships as one vendored file (``model.py``) holding every component, and
upstream inference is a method on the transformer. This module splits it into the
three independently loadable pieces the loader exposes as variants -- T5-XL text
encoder, the 2B transformer, and the BSQ-VAE decoder -- each with a wrapper that
takes positional tensors and returns a bare tensor, plus its Megatron shard spec
for the shared ``(1, num_devices)`` / ``(None, "model")`` mesh.
"""

from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from . import model as _m

# ── Checkpoints ───────────────────────────────────────────────────────
REPO_ID = "FoundationVision/Infinity"
TRANSFORMER_FILENAME = "infinity_2b_reg.pth"
VAE_FILENAME = "infinity_vae_d32reg.pth"
TEXT_ENCODER_HF_ID = "google/flan-t5-xl"

# Weight dtype for TT execution (bf16 fits the 1M schedule in DRAM).
DTYPE = torch.bfloat16

# Resolution preset and model dims the run args are built from.
PN = "1M"
TEXT_CHANNELS = 2048
VAE_TYPE = 32

# ── Mesh ──────────────────────────────────────────────────────────────
# Every component shares one mesh so the whole pipeline runs on it: the
# ``model`` axis (index 1) carries the tensor parallelism and the size-1 axis is
# named ``None`` so no partition spec ever references it.
MESH_NAMES = (None, "model")
MESH_SHAPES = {4: (1, 4), 8: (1, 8)}
# Attention heads of the transformer -- the ``model`` axis must divide these.
TRANSFORMER_NUM_HEADS = 16


def build_run_args():
    """SimpleNamespace mirroring the args ``run_infinity``'s loaders read."""
    from types import SimpleNamespace

    return SimpleNamespace(
        model_type="infinity_2b",
        model_path=hf_hub_download(repo_id=REPO_ID, filename=TRANSFORMER_FILENAME),
        checkpoint_type="torch",
        enable_model_cache=0,
        pn=PN,
        use_bit_label=1,
        add_lvl_embeding_only_first_block=1,
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        text_channels=TEXT_CHANNELS,
        apply_spatial_patchify=0,
        use_flex_attn=0,
        bf16=0,
        vae_type=VAE_TYPE,
        vae_path=hf_hub_download(repo_id=REPO_ID, filename=VAE_FILENAME),
        text_encoder_ckpt=TEXT_ENCODER_HF_ID,
    )


# ── Component loaders ─────────────────────────────────────────────────


def load_tokenizer_and_encoder():
    """T5-XL tokenizer + encoder exactly as ``model.load_tokenizer`` builds them
    (fp16, on the best local device) -- the reference pair ``build_forward_inputs``
    encodes with."""
    return _m.load_tokenizer(t5_path=TEXT_ENCODER_HF_ID)


def load_tokenizer():
    """T5-XL tokenizer (``model_max_length`` 512, as Infinity configures it)."""
    tokenizer, _ = load_tokenizer_and_encoder()
    return tokenizer


def load_text_encoder(dtype: Optional[torch.dtype] = None):
    """T5-XL encoder, wrapped to return a bare ``last_hidden_state``."""
    _, encoder = _m.load_tokenizer(t5_path=TEXT_ENCODER_HF_ID)
    # _m.load_tokenizer hands back an fp16 encoder on the best local device;
    # re-cast for TT and keep it on CPU until the caller places it.
    encoder = encoder.to("cpu", dtype or DTYPE).eval()
    return T5TextEncoderWrapper(encoder).eval()


def load_vae(dtype: Optional[torch.dtype] = None):
    """Full BSQ-VAE (fp32).

    Returned whole, not decoder-only: the per-scale bit-label bookkeeping in the
    pipeline runs on ``vae.quantizer`` on CPU while only ``vae.decoder`` is
    placed on TT. ``dtype`` is accepted for symmetry and applied to the decoder
    only, by :class:`VAEDecoderWrapper`.
    """
    return _m.load_visual_tokenizer(build_run_args()).eval()


def load_transformer(vae, dtype: Optional[torch.dtype] = None):
    """Infinity 2B transformer, with the training-time conditioning drop disabled.

    Needs a VAE instance: ``Infinity.__init__`` reads ``embed_dim``,
    ``vocab_size`` and the bit-label mask (``quantizer.lfq.mask``) off it.

    ``cond_drop_rate`` defaults to 0.1 and is live in ``Infinity.forward``:
    ``if random.random() < self.cond_drop_rate`` replaces the prompt conditioning
    with ``cfg_uncond``. That is classifier-free-guidance training augmentation,
    and at inference it means one forward in ten silently ignores the prompt.
    It also makes the forward non-deterministic *across calls* -- which a golden
    comparison notices, since the device and CPU runs each roll their own value
    (18% of the time exactly one of them drops, and the PCC collapses to ~0.3).
    """
    model = _m.load_transformer(vae, build_run_args()).eval()
    model.cond_drop_rate = 0.0
    return model.to(dtype or DTYPE)


# ── Wrappers: positional tensors in, bare tensor out ──────────────────


class T5TextEncoderWrapper(nn.Module):
    """T5-XL encoder returning ``last_hidden_state`` instead of a ModelOutput."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state


class _ShardableGroupNorm(nn.GroupNorm):
    """GroupNorm with the grouping made explicit, so it survives channel sharding.

    ``nn.GroupNorm`` lowers to the ``ttir.group_norm`` composite, whose
    ``num_groups`` attribute is not rescaled when Shardy shards the channel dim.
    On 8 devices the decoder's 128-channel norms arrive as C=16 still carrying
    num_groups=32 and tt-mlir rejects them outright ("channel dimension (dim 1)
    must be divisible by num_groups"); the 512- and 256-channel norms do divide,
    so they compile but group wrongly -- 64 local channels normalized as 32
    groups of 2 rather than 4 groups of 16. Fixed upstream by
    tenstorrent/tt-mlir#9174 and #9218, neither of which is in the pinned
    toolchain.

    Reshaping to ``[B, G, C // G, *spatial]`` and reducing over everything after
    the group axis states the grouping in the IR instead of in an attribute: the
    partitioner splits whole groups (the shard factor divides G) and every device
    computes complete, correct statistics with no collective. Statistics are
    taken in fp32, as ``nn.GroupNorm`` does internally.

    Swapped in per instance as a real subclass, not a ``forward`` assignment:
    Dynamo resolves a submodule call through ``type(mod).forward`` and would drop
    an instance-level override from the compiled graph. Safe to delete once the
    toolchain carries the rescale.
    """

    def forward(self, x):
        shape = x.shape
        B, C, spatial = shape[0], shape[1], shape[2:]
        xg = x.float().reshape(B, self.num_groups, C // self.num_groups, *spatial)
        dims = tuple(range(2, xg.ndim))
        mu = xg.mean(dims, keepdim=True)
        var = (xg - mu).pow(2).mean(dims, keepdim=True)
        y = ((xg - mu) * torch.rsqrt(var + self.eps)).reshape(shape)
        affine_shape = (1, C) + (1,) * len(spatial)
        if self.weight is not None:
            y = y * self.weight.reshape(affine_shape).float()
        if self.bias is not None:
            y = y + self.bias.reshape(affine_shape).float()
        return y.to(x.dtype)


class VAEDecoderWrapper(nn.Module):
    """BSQ-VAE decoder: latent ``z`` -> RGB in [-1, 1] (``AutoEncoder.decode``).

    Holds ``vae.decoder`` only, so placing this wrapper on TT moves the decoder
    while the parent ``AutoEncoder`` -- in particular ``quantizer`` -- stays on
    CPU for the per-scale code bookkeeping.

    Every GroupNorm in the decoder is retyped to :class:`_ShardableGroupNorm`;
    see there for why. The swap keeps the same parameter objects and names, so
    the shard spec (keyed by parameter) is unaffected, and it applies to the CPU
    reference too -- both sides of a golden comparison run the same math.
    """

    def __init__(self, vae, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.decoder = vae.decoder.to(dtype or DTYPE)
        for mod in self.decoder.modules():
            if type(mod) is nn.GroupNorm:
                mod.__class__ = _ShardableGroupNorm

    def forward(self, z):
        return torch.clamp(self.decoder(z), min=-1, max=1)


# ── Shard specs (Megatron column -> row on the "model" axis) ───────────


def shard_text_encoder_specs(model):
    """Megatron column->row spec for the T5-XL encoder.

    Per block: q/k/v column-parallel (split the 32 heads), ``o`` row-parallel;
    ``wi_0``/``wi_1`` column-parallel, ``wo`` row-parallel -- two all-reduces per
    block. T5 has no attention/FFN biases. The RMS norms and block 0's
    ``relative_attention_bias`` are left replicated; the partitioner slices the
    bias to match the sharded heads.

    Args:
        model: :class:`T5TextEncoderWrapper` or a bare ``T5EncoderModel``.

    Returns:
        Dict[torch.Tensor, tuple]: parameter -> partition spec.
    """
    encoder = getattr(model, "encoder", model)
    # Unwrap T5EncoderModel -> T5Stack.
    stack = getattr(encoder, "encoder", encoder)
    specs = {}
    for block in stack.block:
        attn = block.layer[0].SelfAttention
        specs[attn.q.weight] = ("model", None)
        specs[attn.k.weight] = ("model", None)
        specs[attn.v.weight] = ("model", None)
        specs[attn.o.weight] = (None, "model")  # row: all-reduce

        ff = block.layer[-1].DenseReluDense
        # flan-t5 is gated-gelu (wi_0/wi_1); plain t5 has a single wi.
        for name in ("wi_0", "wi_1", "wi"):
            proj = getattr(ff, name, None)
            if proj is not None:
                specs[proj.weight] = ("model", None)
        specs[ff.wo.weight] = (None, "model")  # row: all-reduce
    return specs


def shard_transformer_specs(model):
    """Megatron head-parallel spec for the Infinity transformer.

    Each of the (unfused) q/k/v projections is column-parallel (split output
    heads on ``model``) and the output ``proj`` is row-parallel (split the
    contraction dim) -- one all-reduce per attention/FFN pair (Megatron-LM,
    arXiv:1909.08053). The projections are unfused in ``model.py``
    (``mat_qkv`` -> ``mat_q/mat_k/mat_v``, ``mat_kv`` -> ``mat_k/mat_v``)
    precisely so a single partition spec can place matching per-head q/k/v on the
    same device -- sharding the fused qkv-major weight directly is numerically
    wrong (PCC ~ -0.18).

    Per-head scale (``scale_mul_1H11``), the concatenated bias buffers, norms,
    lvl/positional embeddings and the tiny head are left replicated; the
    partitioner slices them to match the sharded heads.

    Args:
        model: the ``Infinity`` transformer instance.

    Returns:
        Dict[torch.Tensor, tuple]: parameter -> partition spec.
    """
    specs = {}
    for block in model.unregistered_blocks:
        # --- self-attention: column-parallel q/k/v, row-parallel proj ---
        sa = block.sa
        specs[sa.mat_q.weight] = ("model", None)
        specs[sa.mat_k.weight] = ("model", None)
        specs[sa.mat_v.weight] = ("model", None)
        specs[sa.proj.weight] = (None, "model")  # row: all-reduce
        if sa.proj.bias is not None:
            specs[sa.proj.bias] = (None,)

        # --- cross-attention: column-parallel q/k/v, row-parallel proj ---
        ca = block.ca
        specs[ca.mat_q.weight] = ("model", None)
        if ca.mat_q.bias is not None:
            specs[ca.mat_q.bias] = ("model",)
        specs[ca.mat_k.weight] = ("model", None)
        specs[ca.mat_v.weight] = ("model", None)
        specs[ca.proj.weight] = (None, "model")  # row: all-reduce
        if ca.proj.bias is not None:
            specs[ca.proj.bias] = (None,)

        # --- FFN (column fc1 -> row fc2) ---
        specs[block.ffn.fc1.weight] = ("model", None)
        if block.ffn.fc1.bias is not None:
            specs[block.ffn.fc1.bias] = ("model",)
        specs[block.ffn.fc2.weight] = (None, "model")
        if block.ffn.fc2.bias is not None:
            specs[block.ffn.fc2.bias] = (None,)
    return specs


def shard_vae_specs(model):
    """Channel-parallel Megatron spec for the BSQ-VAE decoder.

    On every ``ResnetBlock``, ``conv1`` is column-parallel on C_out and ``conv2``
    row-parallel on C_in, so each block is replicated in / replicated out with a
    single all-reduce -- 17 blocks (``mid.block_1``, ``mid.block_2`` and 3 per
    each of the 5 up levels), 17 all-reduces. ``conv_in``, the ``Upsample``
    convs, ``nin_shortcut``, ``norm_out`` and ``conv_out`` stay replicated, as
    does every GroupNorm.

    NOTE: ``norm2`` sits between the sharded conv pair, so it normalizes a
    channel-sharded activation. That is only correct because
    :class:`_ShardableGroupNorm` states the grouping in the IR; the stock
    ``nn.GroupNorm`` composite carries ``num_groups`` as an attribute the pinned
    tt-mlir does not rescale when the channels are sharded.

    Args:
        model: :class:`VAEDecoderWrapper`, a bare ``Decoder``, or an
            ``AutoEncoder``.

    Returns:
        Dict[torch.Tensor, tuple]: parameter -> partition spec.
    """
    decoder = getattr(model, "decoder", model)
    specs = {}
    for block in decoder.modules():
        if not isinstance(block, _m.ResnetBlock):
            continue
        # ``Conv`` wraps the real nn.Conv2d as ``.conv``.
        c1, c2 = block.conv1.conv, block.conv2.conv
        specs[c1.weight] = ("model", None, None, None)  # column: split C_out
        if c1.bias is not None:
            specs[c1.bias] = ("model",)
        specs[c2.weight] = (None, "model", None, None)  # row: all-reduce
        if c2.bias is not None:
            specs[c2.bias] = (None,)
    return specs


# ── Component inputs ──────────────────────────────────────────────────


def load_text_encoder_inputs(
    tokenizer,
    dtype_override: Optional[torch.dtype] = None,
    prompt: Optional[str] = None,
):
    """``[input_ids, attention_mask]`` (1, 512) int64 for the T5-XL encoder.

    Padded to 512 exactly as ``model.encode_prompt`` does, so the component test
    sees the sequence length the pipeline actually runs.
    """
    prompt = prompt or "A fantasy landscape with mountains and rivers"
    tokens = tokenizer(
        text=[prompt],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return [tokens.input_ids, tokens.attention_mask]


def load_vae_inputs(dtype_override: Optional[torch.dtype] = None):
    """``[z]`` (1, 32, 64, 64) -- the 1M schedule's final latent, decoded to 1024x1024.

    Matches what the pipeline feeds ``vae.decode``: the accumulated codes at the
    last scale of ``dynamic_resolution_h_w[1.0]["1M"]``, with the singleton
    temporal dim squeezed out.
    """
    _, h, w = _m.dynamic_resolution_h_w[1.000][PN]["scales"][-1]
    z = torch.randn(1, VAE_TYPE, h, w)
    return [z.to(dtype_override or DTYPE)]


def build_forward_inputs(
    tokenizer,
    text_encoder,
    vae,
    pn: str = "1M",
    h_div_w: float = 1.000,
    batch_size: int = 1,
    prompt: Optional[str] = None,
    dtype_override: Optional[torch.dtype] = None,
):
    """Build a single forward-pass input dict for ``Infinity.forward``.

    Args:
        tokenizer: T5 tokenizer (from ``model.load_tokenizer``).
        text_encoder: T5EncoderModel (from ``model.load_tokenizer``).
        vae: BSQ AutoEncoder (from ``model.load_visual_tokenizer``); only
            its ``embed_dim`` attribute is read here.
        pn: Resolution preset key into ``dynamic_resolution_h_w``
            ("0.06M", "0.25M", "0.60M", or "1M"). Default "1M" (~1024x1024).
        h_div_w: Aspect-ratio key into ``dynamic_resolution_h_w``. Default 1.000.
        batch_size: Number of prompt copies to encode.
        prompt: Text prompt; defaults to a fixed deterministic string.
        dtype_override: Optional ``torch.dtype`` to cast the tensor inputs.

    Returns:
        dict: ``{"label_B_or_BLT": (kv_compact, lens, cu_seqlens_k,
            max_seqlen_k), "x_BLC_wo_prefix": tensor, "scale_schedule": [...]}``.
    """
    prompt = prompt or "A fantasy landscape with mountains and rivers"

    kv_list, lens_total = [], []
    for _ in range(batch_size):
        kv, lens, _, _ = _m.encode_prompt(tokenizer, text_encoder, prompt)
        kv_list.append(kv)
        lens_total.extend(lens)
    kv_compact = torch.cat(kv_list, dim=0)
    cu_seqlens_k = torch.tensor(
        [0] + list(torch.tensor(lens_total).cumsum(0).tolist()),
        dtype=torch.int32,
        device=kv_compact.device,
    )
    max_seqlen_k = max(lens_total)

    sched = _m.dynamic_resolution_h_w[h_div_w][pn]["scales"]
    scale_schedule = [(1, h, w) for (_, h, w) in sched]
    total_visual_tokens = sum(pt * ph * pw for pt, ph, pw in scale_schedule)
    # Inside ``Infinity.forward`` the SOS token replaces the first scale, so
    # the model concats sos (1 token) + word_embed(x_BLC_wo_prefix) and the
    # resulting ``l_end`` must equal ``sum(scale_schedule)``.  That means
    # ``x_BLC_wo_prefix`` length = total_visual_tokens - first_scale_count.
    # word_embed is ``nn.Linear(d_vae, C)`` so the last dim is d_vae
    # (vae.embed_dim = codebook_dim = vae_type, e.g. 32).
    d_vae = vae.embed_dim
    first_scale_count = (
        scale_schedule[0][0] * scale_schedule[0][1] * scale_schedule[0][2]
    )
    # Random, not zeros: ``norm0_ve`` is a LayerNorm over d_vae, so an all-zero
    # token normalizes to its bias and ``word_embed`` maps EVERY visual position
    # to the same vector -- the packed sequence collapses to one distinct row per
    # scale. A golden comparison on that input measures almost nothing: PCC
    # subtracts the mean, the large constant component cancels, and what is left
    # is the small positional variation where bf16 noise lives. Seeded from a
    # local generator so the inputs are reproducible without disturbing global
    # RNG. The magnitude is not important -- norm0_ve normalizes it away.
    gen = torch.Generator().manual_seed(42)
    x_BLC_wo_prefix = torch.randn(
        batch_size,
        total_visual_tokens - first_scale_count,
        d_vae,
        generator=gen,
    )

    if dtype_override is not None:
        kv_compact = kv_compact.to(dtype_override)
        x_BLC_wo_prefix = x_BLC_wo_prefix.to(dtype_override)

    return {
        "label_B_or_BLT": (kv_compact, lens_total, cu_seqlens_k, max_seqlen_k),
        "x_BLC_wo_prefix": x_BLC_wo_prefix,
        "scale_schedule": scale_schedule,
    }
