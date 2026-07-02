# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LongCat-Video per-component loader.

meituan-longcat/LongCat-Video is a ~20B text/image-to-video diffusion pipeline
(arXiv:2510.22200). It is NOT a standard diffusers pipeline -- the model_index
is empty and the modeling code lives in the LongCat-Video GitHub repo
(github.com/meituan-longcat/LongCat-Video, package `longcat_video`). Components:

  tokenizer    -> AutoTokenizer (T5Tokenizer, spiece)              no parameters
  text_encoder -> UMT5EncoderModel (google/umt5-xxl)               params ~5.7 B
  vae          -> AutoencoderKLWan (Wan 3D causal VAE, 16-ch z)    params ~0.25 B
  scheduler    -> FlowMatchEulerDiscreteScheduler                  no parameters
  dit          -> LongCatVideoTransformer3DModel  (KEY component)  params ~13.6 B
                  depth=48, hidden=4096, 32 heads, patch (1,2,2),
                  caption_channels=4096, single-stream blocks, 3D RoPE.

Each variant scaffolds one component as a tensors-only torch.nn.Module the
runner can compile + PCC-compare in isolation. The full pipeline (scheduler
loop, latent glue) stays in host Python -- see the composite test.

Two adaptations are required to run the DiT off-CUDA:
  * cp_split_hw=[1,1] (single device; the repo's context-parallel path uses
    torch.distributed all-to-all and is bypassed when not distributed).
  * The repo attention has NO torch fallback -- it dispatches only to
    flash-attn2/3 / xformers / block-sparse and otherwise raises. flash-attn
    is CUDA-only, so we monkeypatch Attention._process_attn to use
    torch.nn.functional.scaled_dot_product_attention (math equivalent; the
    flash path is just a fused SDPA). See _patch_attention_sdpa().

The repo is not pip-installable; _ensure_longcat_repo() shallow-clones it to a
cache dir and puts it on sys.path on first use (pin in LONGCAT_VIDEO_COMMIT).
"""

import os
import subprocess
import sys
from typing import Optional

import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

# ---- repo ------------------------------------------------------------------
LONGCAT_VIDEO_REPO_ID = "meituan-longcat/LongCat-Video"
LONGCAT_VIDEO_GIT = "https://github.com/meituan-longcat/LongCat-Video.git"
# Pin the modeling code; bump deliberately. (HEAD of main as of bringup.)
LONGCAT_VIDEO_COMMIT = None  # None -> latest main (shallow clone)

DTYPE = torch.bfloat16

# ---- native t2v geometry (pipeline defaults: 480x832, 93 frames) -----------
# vae spatial downsample 8, temporal downsample 4.
NATIVE_HEIGHT = 480
NATIVE_WIDTH = 832
NATIVE_FRAMES = 93
VAE_SPATIAL = 8
VAE_TEMPORAL = 4
LATENT_C = 16
LATENT_T = (NATIVE_FRAMES - 1) // VAE_TEMPORAL + 1  # 24
LATENT_H = NATIVE_HEIGHT // VAE_SPATIAL  # 60
LATENT_W = NATIVE_WIDTH // VAE_SPATIAL  # 104

# text encoder
TEXT_MAX_LEN = 512
CAPTION_CHANNELS = 4096
TE_VOCAB = 256384


def _ensure_longcat_repo():
    """Shallow-clone the LongCat-Video repo to the HF cache and put it on
    sys.path so `import longcat_video` works. Idempotent."""
    cache_root = os.environ.get(
        "LONGCAT_VIDEO_HOME",
        os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "longcat_video_repo",
        ),
    )
    repo_dir = os.path.join(cache_root, "LongCat-Video")
    # The repo's `longcat_video` package has no top-level __init__.py (implicit
    # namespace package); use a stable module file as the presence sentinel.
    sentinel = os.path.join(
        repo_dir, "longcat_video", "modules", "longcat_video_dit.py"
    )
    if not os.path.exists(sentinel):
        os.makedirs(cache_root, exist_ok=True)
        cmd = ["git", "clone", "--depth", "1", LONGCAT_VIDEO_GIT, repo_dir]
        subprocess.run(cmd, check=True)
        if LONGCAT_VIDEO_COMMIT:
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", LONGCAT_VIDEO_COMMIT],
                cwd=repo_dir,
                check=True,
            )
            subprocess.run(
                ["git", "checkout", LONGCAT_VIDEO_COMMIT], cwd=repo_dir, check=True
            )
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    return repo_dir


def _patch_attention_sdpa():
    """Replace the repo attention's flash-attn-only dispatch with SDPA so it
    runs on CPU / XLA. _process_attn takes q,k,v in [B,H,S,D] and returns the
    same layout -- exactly SDPA's contract."""
    from longcat_video.modules import attention as _attn

    def _sdpa_process_attn(self, q, k, v, shape):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=self.scale
        )

    _attn.Attention._process_attn = _sdpa_process_attn

    def _sdpa_process_cross_attn(self, x, cond, kv_seqlen):
        # Cross-attention from latent tokens (x) to text tokens (cond). The repo
        # packs everything into a single varlen sequence; with no text padding
        # (encoder_attention_mask=None -> full attention) this is plain SDPA.
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_linear(x).view(1, -1, H, D)
        if hasattr(self, "k_linear"):  # kv split applied (tensor parallel)
            k = self.k_linear(cond).view(1, -1, H, D)
            v = self.v_linear(cond).view(1, -1, H, D)
        else:
            kv = self.kv_linear(cond).view(1, -1, 2, H, D)
            k, v = kv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q.transpose(1, 2)  # [1, H, Sq, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, -1, C)
        return self.proj(out)

    _attn.MultiHeadCrossAttention._process_cross_attn = _sdpa_process_cross_attn


def _split_attention_qkv(dit):
    """Split each self-attention's combined ``qkv`` Linear(dim, 3*dim) into three
    separate head-parallel projections (q_proj / k_proj / v_proj), each
    Linear(dim, dim), and rebind ``Attention.forward`` to use them.

    Why: the combined qkv weight is laid out as (3, heads, head_dim) along its
    output, so a *flat* column shard for tensor parallelism crosses the q/k/v
    boundary and misaligns -- forcing qkv (and therefore the whole attention
    activation) to stay REPLICATED. At native t2v seq=37440 the replicated
    attention reshape needs a ~39 GB DRAM buffer per chip and OOMs. Splitting
    into three independent Linears lets each be column-sharded on heads
    ("model", None) and ``proj`` row-sharded (None, "model") -- the standard
    Megatron attention TP -- so the attention activation is divided across chips.

    The original combined ``qkv`` weight is deleted after copying, so per-chip
    weight memory is unchanged (three dim x dim == one dim x 3*dim)."""
    import torch.nn as nn
    from longcat_video.modules import attention as _attn

    for blk in dit.blocks:
        attn = blk.attn
        dim = attn.num_heads * attn.head_dim
        w = attn.qkv.weight.data  # [3*dim, dim], rows ordered q | k | v
        b = attn.qkv.bias.data if attn.qkv.bias is not None else None
        dev, dt = w.device, w.dtype
        for i, name in enumerate(("q_proj", "k_proj", "v_proj")):
            lin = nn.Linear(dim, dim, bias=b is not None).to(device=dev, dtype=dt)
            lin.weight.data.copy_(w[i * dim : (i + 1) * dim])
            if b is not None:
                lin.bias.data.copy_(b[i * dim : (i + 1) * dim])
            setattr(attn, name, lin)
        del attn.qkv  # free the combined weight (avoid doubling attn weights)

        # Cross-attention: split the combined kv_linear into k_linear / v_linear
        # so cross-attn weights can be head-sharded too. Cross-attn weights are
        # ~6 GB replicated across the 48 blocks; sharding them frees the DRAM the
        # self-attn activation needs to fit at native seq.
        ca = blk.cross_attn
        cdim = ca.num_heads * ca.head_dim
        kvw = ca.kv_linear.weight.data  # [2*dim, dim], rows ordered k | v
        kvb = ca.kv_linear.bias.data if ca.kv_linear.bias is not None else None
        for i, name in enumerate(("k_linear", "v_linear")):
            lin = nn.Linear(cdim, cdim, bias=kvb is not None).to(device=dev, dtype=dt)
            lin.weight.data.copy_(kvw[i * cdim : (i + 1) * cdim])
            if kvb is not None:
                lin.bias.data.copy_(kvb[i * cdim : (i + 1) * cdim])
            setattr(ca, name, lin)
        del ca.kv_linear

    def _split_forward(self, x, shape=None, num_cond_latents=None, return_kv=False):
        # Mirrors the repo's Attention.forward exactly, but builds q/k/v from the
        # three split projections instead of one combined reshape.
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, N, H, D).permute(0, 2, 1, 3)  # [B,H,N,D]
        k = self.k_proj(x).view(B, N, H, D).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, N, H, D).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)
        if return_kv:
            k_cache, v_cache = k.clone(), v.clone()
        q, k = self.rope_3d(q, k, shape)
        if num_cond_latents is not None and num_cond_latents > 0:
            nc = num_cond_latents * (N // shape[0])
            q_cond = q[:, :, :nc].contiguous()
            k_cond = k[:, :, :nc].contiguous()
            v_cond = v[:, :, :nc].contiguous()
            x_cond = self._process_attn(q_cond, k_cond, v_cond, shape)
            q_noise = q[:, :, nc:].contiguous()
            x_noise = self._process_attn(q_noise, k, v, shape)
            x = torch.cat([x_cond, x_noise], dim=2).contiguous()
        else:
            x = self._process_attn(q, k, v, shape)
        x = x.transpose(1, 2).reshape(B, N, C)  # [B,H,N,D] -> [B,N,C]
        x = self.proj(x)
        return (x, (k_cache, v_cache)) if return_kv else x

    _attn.Attention.forward = _split_forward


def _fp32_modulation(dit):
    """The adaLN / timestep *modulation* path runs in fp32 -- the repo relies on
    `amp.autocast(device_type='cuda', fp32)` for this, which no-ops off CUDA
    (CPU / XLA). Cast exactly those submodules to fp32 so the fp32 timestep
    embedding `t` flows correctly. The bf16 residual stream and attention are
    preserved: each block casts the gated residual back to bf16 (x.to(x_dtype)),
    and modulate_fp32 returns bf16, so the qkv/proj/ffn linears stay bf16."""
    dit.t_embedder.float()
    dit.final_layer.adaLN_modulation.float()
    for blk in dit.blocks:
        blk.adaLN_modulation.float()


# ---- tensors-only wrappers -------------------------------------------------
class _TextEncoderWrapper(torch.nn.Module):
    """UMT5 encoder -> last_hidden_state [B, seq, 4096] (the prompt embedding
    the pipeline feeds to the DiT)."""

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state


class _DiTWrapper(torch.nn.Module):
    """LongCatVideoTransformer3DModel -> noise prediction. encoder_attention_mask
    is all-ones (no padding) so it is omitted -> the simpler no-mask path."""

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        return self.dit(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
        )


class _VAEDecoderWrapper(torch.nn.Module):
    """AutoencoderKLWan.decode(latent) -> video [B, 3, T, H, W]."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent, return_dict=False)[0]


class ModelVariant(StrEnum):
    """Loadable components of the LongCat-Video pipeline."""

    TEXT_ENCODER = "text_encoder"
    DIT = "dit"
    VAE = "vae"


class ModelLoader(ForgeModel):
    """Per-component loader for LongCat-Video. load_model() returns just the
    requested component (wrapped to a tensors-only forward); load_inputs()
    builds synthetic tensors at native t2v (480x832, 93 frames) shapes."""

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(
            pretrained_model_name=LONGCAT_VIDEO_REPO_ID
        ),
        ModelVariant.DIT: ModelConfig(pretrained_model_name=LONGCAT_VIDEO_REPO_ID),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=LONGCAT_VIDEO_REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.DIT

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant == ModelVariant.TEXT_ENCODER
            else ModelTask.CONDITIONAL_GENERATION
        )
        return ModelInfo(
            model="LongCatVideo",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    # ---- multi-chip (Step 7.5) ------------------------------------------------
    # Mesh axes: (None, "model"); only "model" is a real shard axis.
    _MESH_SHAPES = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (1, 8)}
    _MESH_NAMES = (None, "model")

    def get_mesh_config(self, num_devices: int):
        """((batch, model) mesh shape, names). The DiT shards across "model";
        text_encoder / vae stay single-chip."""
        if self._variant != ModelVariant.DIT:
            return (1, 1), self._MESH_NAMES
        if num_devices not in self._MESH_SHAPES:
            raise ValueError(f"Unsupported device count {num_devices}")
        return self._MESH_SHAPES[num_devices], self._MESH_NAMES

    def load_shard_spec(self, model):
        """tensor -> partition_spec for the DiT (Pattern B, tensor-parallel).

        SwiGLU FFN is the bulk of each block (~2/3 of params) and shards cleanly
        Megatron-style: w1/w3 (gate/up) column-parallel ("model", None), w2
        (down) row-parallel (None, "model").

        Attention is sharded too **when the qkv split has been applied**
        (see _split_attention_qkv, enabled by load_model(split_attention=True)):
        q_proj/k_proj/v_proj column-parallel on heads ("model", None) with their
        biases sharded ("model",), and proj row-parallel (None, "model"). This is
        required for native t2v seq=37440 -- with attention replicated the
        per-step reshape needs a ~39 GB DRAM buffer and OOMs. If the split was
        not applied, the combined qkv Linear(dim, 3*dim) cannot be flat-column
        sharded (its (3, heads, head_dim) output view misaligns), so attention is
        left replicated (fits only at reduced seq)."""
        dit = getattr(model, "dit", model)
        specs = {}
        for blk in dit.blocks:
            ffn = blk.ffn
            specs[ffn.w1.weight] = ("model", None)  # gate (column)
            specs[ffn.w3.weight] = ("model", None)  # up   (column)
            specs[ffn.w2.weight] = (None, "model")  # down (row)

            attn = blk.attn
            if hasattr(attn, "q_proj"):  # qkv split applied -> shard attention
                for name in ("q_proj", "k_proj", "v_proj"):
                    lin = getattr(attn, name)
                    specs[lin.weight] = ("model", None)  # column (heads)
                    if lin.bias is not None:
                        specs[lin.bias] = ("model",)
                specs[attn.proj.weight] = (None, "model")  # row

            ca = blk.cross_attn
            if hasattr(ca, "k_linear"):  # cross-attn kv split -> shard cross-attn
                for name in ("q_linear", "k_linear", "v_linear"):
                    lin = getattr(ca, name)
                    specs[lin.weight] = ("model", None)  # column (heads)
                    if lin.bias is not None:
                        specs[lin.bias] = ("model",)
                specs[ca.proj.weight] = (None, "model")  # row
        return specs

    def _checkpoint_dir(self):
        """Local snapshot dir of the HF checkpoint (subfolders dit/, vae/, ...)."""
        from huggingface_hub import snapshot_download

        return snapshot_download(
            LONGCAT_VIDEO_REPO_ID,
            allow_patterns=[
                "dit/*",
                "text_encoder/*",
                "vae/*",
                "scheduler/*",
                "tokenizer/*",
                "model_index.json",
                "config.json",
            ],
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        split_attention: bool = False,
        **kwargs,
    ):
        dtype = dtype_override if dtype_override is not None else DTYPE
        ckpt = self._checkpoint_dir()

        if self._variant == ModelVariant.TEXT_ENCODER:
            from transformers import UMT5EncoderModel

            te = UMT5EncoderModel.from_pretrained(
                ckpt, subfolder="text_encoder", torch_dtype=dtype
            )
            return _TextEncoderWrapper(te.eval())

        if self._variant == ModelVariant.VAE:
            _ensure_longcat_repo()
            from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan

            vae = AutoencoderKLWan.from_pretrained(
                ckpt, subfolder="vae", torch_dtype=dtype
            )
            return _VAEDecoderWrapper(vae.eval())

        if self._variant == ModelVariant.DIT:
            _ensure_longcat_repo()
            _patch_attention_sdpa()
            from longcat_video.modules.longcat_video_dit import (
                LongCatVideoTransformer3DModel,
            )

            dit = LongCatVideoTransformer3DModel.from_pretrained(
                ckpt,
                subfolder="dit",
                cp_split_hw=[1, 1],  # single device: no context parallel
                torch_dtype=dtype,
            )
            if split_attention:
                _split_attention_qkv(dit)
            if dtype != torch.float32:
                _fp32_modulation(dit)
            return _DiTWrapper(dit.eval())

        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        **kwargs,
    ):
        dtype = dtype_override if dtype_override is not None else DTYPE
        B = batch_size

        if self._variant == ModelVariant.TEXT_ENCODER:
            input_ids = torch.randint(0, TE_VOCAB, (B, TEXT_MAX_LEN), dtype=torch.long)
            attention_mask = torch.ones(B, TEXT_MAX_LEN, dtype=torch.long)
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.DIT:
            hidden_states = torch.randn(
                B, LATENT_C, LATENT_T, LATENT_H, LATENT_W, dtype=dtype
            )
            timestep = torch.ones(B, dtype=dtype) * 999.0
            # CaptionEmbedder expects [B, 1, N_token, C] (pipeline shapes the T5
            # prompt embeds to a singleton "modality" dim).
            encoder_hidden_states = torch.randn(
                B, 1, TEXT_MAX_LEN, CAPTION_CHANNELS, dtype=dtype
            )
            return [hidden_states, timestep, encoder_hidden_states]

        if self._variant == ModelVariant.VAE:
            latent = torch.randn(
                B, LATENT_C, LATENT_T, LATENT_H, LATENT_W, dtype=dtype
            )
            return [latent]

        raise ValueError(f"Unknown variant: {self._variant}")
