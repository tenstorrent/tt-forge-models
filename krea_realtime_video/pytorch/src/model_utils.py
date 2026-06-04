# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Component loaders, wrappers, and shard specs for Krea Realtime Video 14B.

Model: krea/krea-realtime-video  (WanModularPipeline)
Components (params from safetensors headers):
  - text_encoder: UMT5EncoderModel  (UMT5-XXL)        ~5.68B   reused from Wan 2.1 14B
  - transformer:  CausalWanModel     (14B video DiT)  ~14.29B  from krea
  - vae:          AutoencoderKLWan   (3D causal VAE)   ~0.13B   reused from Wan 2.1 14B

All I/O shapes/dtypes here were captured from one real CPU forward per
component; see .claude/bringup/krea_realtime_video/io_spec.json.
"""

import torch

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

# text_encoder, vae, tokenizer, scheduler come from the Wan 2.1 14B base repo.
# Only the transformer weights come from krea (modular_model_index.json).
KREA_REPO_ID = "krea/krea-realtime-video"
WAN_REPO_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Inference shape constants (captured)
# ---------------------------------------------------------------------------

HEIGHT = 480
WIDTH = 832
NUM_BLOCKS = 1
MAX_SEQ_LEN = 512  # text token length (transformer text_len)

LATENT_H = HEIGHT // 8  # 60
LATENT_W = WIDTH // 8  # 104
NUM_FRAMES_PER_BLOCK = 3
NUM_LATENT_FRAMES = NUM_BLOCKS * NUM_FRAMES_PER_BLOCK  # 3
NUM_CHANNELS_LATENTS = 16
TEXT_EMBED_DIM = 4096

# KV-cache config (self-attention over video frames; CausVid-style causal cache)
KV_CACHE_NUM_FRAMES = 3
FRAME_SEQ_LENGTH = 1560
SEQ_LENGTH = 32760
LOCAL_ATTN_SIZE = KV_CACHE_NUM_FRAMES + NUM_FRAMES_PER_BLOCK  # 6
KV_CACHE_SIZE = LOCAL_ATTN_SIZE * FRAME_SEQ_LENGTH  # 9360

# UMT5 vocabulary size (Embedding(256384, 4096))
UMT5_VOCAB_SIZE = 256384

# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------


def load_text_encoder(pretrained_model_name: str, dtype: torch.dtype):
    """Load UMT5EncoderModel from the text_encoder subfolder (Wan base repo)."""
    from transformers import UMT5EncoderModel

    return UMT5EncoderModel.from_pretrained(
        pretrained_model_name,
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


def load_transformer(pretrained_model_name: str, dtype: torch.dtype):
    """Load CausalWanModel from the transformer subfolder (krea repo).

    CausalWanModel is a custom architecture defined in krea/krea-realtime-video
    (transformer/causal_model.py). diffusers.AutoModel with
    trust_remote_code=True downloads and runs the custom model code to resolve
    the class.
    """
    from diffusers import AutoModel

    return AutoModel.from_pretrained(
        pretrained_model_name,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()


def load_vae(pretrained_model_name: str, dtype: torch.dtype):
    """Load AutoencoderKLWan from the vae subfolder (Wan base repo)."""
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=dtype,
        device_map="cpu",
    ).eval()


# ---------------------------------------------------------------------------
# Sinusoidal embedding fix
# ---------------------------------------------------------------------------


def fixed_sinusoidal_embedding_1d(dim, position):
    """Device-agnostic replacement for Krea's CUDA-hardcoded sinusoidal embedding.

    Upstream (transformer/model.py:33) uses ``device=torch.cuda.current_device()``
    which crashes on CPU and TT. We replace it with ``device=position.device``.
    Ref: https://github.com/tenstorrent/tt-xla/issues/4464
    """
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(
            10000,
            -torch.arange(half, device=position.device, dtype=torch.float64).div(half),
        ),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------


class CausalWanWrapper(torch.nn.Module):
    """Simplify CausalWanModel forward to (x, t, context) -> noise_pred.

    The raw model (CausalWanModel._forward_inference) requires kv_cache,
    crossattn_cache, an integer seq_len, and several positional args on every
    call, and consumes x/context as *lists* of tensors. This wrapper:
      - rebuilds all per-block caches fresh each forward (stateless)
      - patches out the CUDA-hardcoded sinusoidal_embedding_1d
      - disables local-attention windowing so a single block runs cleanly
    """

    def __init__(self, transformer, num_frames_per_block: int = NUM_FRAMES_PER_BLOCK):
        super().__init__()
        self.transformer = transformer

        # Replace the CUDA-only embedding function in the model's global scope.
        transformer.forward.__globals__["sinusoidal_embedding_1d"] = (
            fixed_sinusoidal_embedding_1d
        )

        for blk in self.transformer.blocks:
            blk.self_attn.local_attn_size = -1
            blk.self_attn.num_frame_per_block = num_frames_per_block

        self._num_blocks = len(self.transformer.blocks)
        self._num_heads = self.transformer.config.num_heads
        self._head_dim = self.transformer.config.dim // self._num_heads

    def _make_caches(self, device, dtype):
        kv_shape = [1, KV_CACHE_SIZE, self._num_heads, self._head_dim]
        ca_shape = [1, MAX_SEQ_LEN, self._num_heads, self._head_dim]
        kv_cache = [
            {
                "k": torch.zeros(kv_shape, dtype=dtype, device=device).contiguous(),
                "v": torch.zeros(kv_shape, dtype=dtype, device=device).contiguous(),
                "global_end_index": 0,
                "local_end_index": 0,
            }
            for _ in range(self._num_blocks)
        ]
        crossattn_cache = [
            {
                "k": torch.zeros(ca_shape, dtype=dtype, device=device).contiguous(),
                "v": torch.zeros(ca_shape, dtype=dtype, device=device).contiguous(),
                "is_init": False,
            }
            for _ in range(self._num_blocks)
        ]
        return kv_cache, crossattn_cache

    def forward(self, x, t, context):
        kv_cache, crossattn_cache = self._make_caches(x.device, x.dtype)
        return self.transformer(
            x=x,
            t=t,
            context=context,
            kv_cache=kv_cache,
            seq_len=SEQ_LENGTH,
            crossattn_cache=crossattn_cache,
            current_start=0,
            cache_start=None,
        )


class VAEDecoderWrapper(torch.nn.Module):
    """Expose only the decoder half of AutoencoderKLWan as (z) -> tensor.

    The default vae(z) runs encode+decode and returns a ModelOutput object.
    This wrapper calls decode directly and unwraps the output to a plain tensor.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


# ---------------------------------------------------------------------------
# SPMD shard specifications (Megatron 1D on a ("batch", "model") 2D mesh)
# ---------------------------------------------------------------------------

# (batch, model) mesh shapes by device count.
MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")


def shard_text_encoder_specs(encoder) -> dict:
    """Shard specs for UMT5EncoderModel.

    Mesh axes: ("batch", "model")
    Column-parallel (q, k, v, wi_0, wi_1): ("model", "batch")
    Row-parallel   (o, wo):                ("batch", "model")
    """
    specs = {encoder.shared.weight: (None, "batch")}

    for block in encoder.encoder.block:
        sa = block.layer[0].SelfAttention
        specs[sa.q.weight] = ("model", "batch")
        specs[sa.k.weight] = ("model", "batch")
        specs[sa.v.weight] = ("model", "batch")
        specs[sa.o.weight] = ("batch", "model")
        specs[block.layer[0].layer_norm.weight] = ("batch",)

        ffn = block.layer[1].DenseReluDense
        specs[ffn.wi_0.weight] = ("model", "batch")
        specs[ffn.wi_1.weight] = ("model", "batch")
        specs[ffn.wo.weight] = ("batch", "model")
        specs[block.layer[1].layer_norm.weight] = ("batch",)

    specs[encoder.encoder.final_layer_norm.weight] = ("batch",)
    return specs


def shard_transformer_specs(transformer) -> dict:
    """Shard specs for CausalWanModel — Megatron 1D on the "model" axis.

    Mesh axes: ("batch", "model"). Only "model" is a real shard axis here;
    "batch" stays replicated (dims that must not shard are marked None, not
    "batch"). This is the Mochi/HunyuanVideo Pattern-A layout.

    Column-parallel (Q, K, V, FFN up):  ("model", None)   bias ("model",)
    Row-parallel    (O, FFN down):      (None, "model")   bias (None,)
    qk RMSNorm (over the flat dim, applied before the head view): ("model",)
      — sharded to match the column-sharded q/k slice; GSPMD inserts the
      variance all-reduce across the "model" axis.

    Everything else is REPLICATED (not in specs): patch_embedding,
    text_embedding, time_embedding, time_projection, head, norm3, modulation.
    These stem tensors feed awkward reshapes — e.g. ``time_projection`` ->
    ``.unflatten(1, (6, dim))`` where 6 is not divisible by the "model" axis
    (4) — which the partitioner cannot propagate sharding through (it produced
    a 4x reshape-element mismatch). They are tiny relative to the 40 blocks, so
    replicating them costs little weight and keeps the hidden state replicated
    between blocks (column->row attention/FFN all-reduce back to full dim).
    """
    specs: dict = {}

    for block in transformer.blocks:
        for attn in (block.self_attn, block.cross_attn):
            specs[attn.q.weight] = ("model", None)
            specs[attn.q.bias] = ("model",)
            specs[attn.k.weight] = ("model", None)
            specs[attn.k.bias] = ("model",)
            specs[attn.v.weight] = ("model", None)
            specs[attn.v.bias] = ("model",)
            specs[attn.o.weight] = (None, "model")
            specs[attn.o.bias] = (None,)
            if hasattr(attn.norm_q, "weight") and attn.norm_q.weight is not None:
                specs[attn.norm_q.weight] = ("model",)
            if hasattr(attn.norm_k, "weight") and attn.norm_k.weight is not None:
                specs[attn.norm_k.weight] = ("model",)

        specs[block.ffn[0].weight] = ("model", None)
        specs[block.ffn[0].bias] = ("model",)
        specs[block.ffn[2].weight] = (None, "model")
        specs[block.ffn[2].bias] = (None,)

    return specs
