# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HunyuanImage 2.1 (Distilled) per-component loader.

Pipeline:  hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers
           (diffusers.HunyuanImagePipeline)

Each ModelVariant maps to one independently-compilable component. ``load_model``
returns *only* that component (wrapped so its forward is tensor-in / tensor-out
for the tt-xla graph runner); ``load_inputs`` synthesises tensors matching the
real shapes captured from a single CPU pipeline pass (see ``_COMPONENT_IO_SPEC``).

  Variant         Component        Class                            params
  --------------  ---------------  -------------------------------  -------
  TEXT_ENCODER    text_encoder     Qwen2_5_VLTextModel (unwrapped)   7.07B
  TEXT_ENCODER_2  text_encoder_2   T5EncoderModel (ByT5 glyph)       0.22B
  TRANSFORMER     transformer      HunyuanImageTransformer2DModel   17.45B
  VAE             vae              AutoencoderKLHunyuanImage         0.41B

Parallelism (see weight_fit):
  - TEXT_ENCODER : single_device on p150 (n150 weight-bound); shard specs
                   provided for 2-chip TP promotion.
  - TEXT_ENCODER_2 / VAE : single_device (n150 + p150); never TP-promoted.
  - TRANSFORMER  : tensor_parallel only — weight-bound on every single-chip
                   arch (bf16 weights 34.9 GiB > p150 27.2 GiB budget).
"""

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

REPO_ID = "hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers"

# Default compute dtype. Tests pass dtype_override (fp32 CPU golden for tight PCC).
DTYPE = torch.float32

# ---------------------------------------------------------------------------
# Shape constants — production resolution (2048 x 2048, single image, batch 1).
# Resolution-independent dims (channels, embed dims, text seq lengths) are taken
# verbatim from the capture; spatial latent dims scale as image // 32.
# ---------------------------------------------------------------------------
IMAGE_H = 2048
IMAGE_W = 2048
VAE_SCALE_FACTOR = 32  # AutoencoderKLHunyuanImage.spatial_compression_ratio
LATENT_H = IMAGE_H // VAE_SCALE_FACTOR  # 64
LATENT_W = IMAGE_W // VAE_SCALE_FACTOR  # 64

NUM_CHANNELS_LATENTS = 64  # transformer.config.in_channels == vae latent_channels

TEXT_EMBED_DIM = 3584  # Qwen2.5-VL hidden size
TEXT_EMBED_2_DIM = 1472  # ByT5 hidden size

TEXT_TOKEN_MAX_LEN = 1034  # tokenizer_max_length (1000) + drop_idx (34)
TEXT_TOKEN_2_MAX_LEN = 128

TRANSFORMER_TEXT_SEQ = 1000  # encoder_hidden_states seq dim (post drop_idx slice)
TRANSFORMER_TEXT_2_SEQ = 128  # encoder_hidden_states_2 seq dim

QWEN_VOCAB_SIZE = 151936  # Qwen2.5-VL text encoder
BYT5_VOCAB_SIZE = 384  # ByT5 text_encoder_2

# ---------------------------------------------------------------------------
# Captured I/O spec (one CPU pass, steps=2, height=width=64 -> latent 2x2).
# Shapes below are reported at production resolution; the capture confirmed the
# component structure (forward signatures, dtypes, channel/embed/seq dims).
# ---------------------------------------------------------------------------
_COMPONENT_IO_SPEC = {
    "text_encoder": {
        "inputs": [
            {
                "name": "input_ids",
                "shape": (1, TEXT_TOKEN_MAX_LEN),
                "dtype": "torch.int64",
            },
            {
                "name": "attention_mask",
                "shape": (1, TEXT_TOKEN_MAX_LEN),
                "dtype": "torch.int64",
            },
        ],
        "outputs": [
            {
                "name": "last_hidden_state",
                "shape": (1, TEXT_TOKEN_MAX_LEN, TEXT_EMBED_DIM),
            }
        ],
        "called_per_step": False,
    },
    "text_encoder_2": {
        "inputs": [
            {
                "name": "input_ids",
                "shape": (1, TEXT_TOKEN_2_MAX_LEN),
                "dtype": "torch.int64",
            },
            {
                "name": "attention_mask",
                "shape": (1, TEXT_TOKEN_2_MAX_LEN),
                "dtype": "torch.float32",
            },
        ],
        "outputs": [
            {
                "name": "last_hidden_state",
                "shape": (1, TEXT_TOKEN_2_MAX_LEN, TEXT_EMBED_2_DIM),
            }
        ],
        "called_per_step": False,
    },
    "transformer": {
        "inputs": [
            {
                "name": "hidden_states",
                "shape": (1, NUM_CHANNELS_LATENTS, LATENT_H, LATENT_W),
                "dtype": "compute",
            },
            {"name": "timestep", "shape": (1,), "dtype": "compute"},
            {"name": "timestep_r", "shape": (1,), "dtype": "compute"},
            {"name": "guidance", "shape": (1,), "dtype": "compute"},
            {
                "name": "encoder_hidden_states",
                "shape": (1, TRANSFORMER_TEXT_SEQ, TEXT_EMBED_DIM),
                "dtype": "compute",
            },
            {
                "name": "encoder_attention_mask",
                "shape": (1, TRANSFORMER_TEXT_SEQ),
                "dtype": "torch.int64",
            },
            {
                "name": "encoder_hidden_states_2",
                "shape": (1, TRANSFORMER_TEXT_2_SEQ, TEXT_EMBED_2_DIM),
                "dtype": "compute",
            },
            {
                "name": "encoder_attention_mask_2",
                "shape": (1, TRANSFORMER_TEXT_2_SEQ),
                "dtype": "torch.int64",
            },
        ],
        "outputs": [
            {"name": "sample", "shape": (1, NUM_CHANNELS_LATENTS, LATENT_H, LATENT_W)}
        ],
        "called_per_step": True,
    },
    "vae": {
        "inputs": [
            {
                "name": "z",
                "shape": (1, NUM_CHANNELS_LATENTS, LATENT_H, LATENT_W),
                "dtype": "compute",
            },
        ],
        "outputs": [{"name": "sample", "shape": (1, 3, IMAGE_H, IMAGE_W)}],
        "called_per_step": False,
    },
}

# ---------------------------------------------------------------------------
# SPMD mesh — ("batch", "model"), Megatron 1D (Pattern A). The "model" axis
# carries all weight sharding; "batch" stays size 1 on the single-image bringup
# path so every (_, "batch") spec is a no-op replicate. A size-2 "batch" axis
# (e.g. 8 -> (2, 4)) forces cross-axis reshards that emit sdy.collective_permute,
# which the Shardy->StableHLO path cannot lower yet (tt-mlir #3370) — keep 1D.
# ---------------------------------------------------------------------------
MESH_SHAPES = {32: (8, 4), 8: (1, 8), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")
# Single-device default; transformer TP promotion uses board IDs from tt-smi.
TT_VISIBLE_DEVICES = "0"


class ModelVariant(StrEnum):
    """Loadable components of the HunyuanImage 2.1 pipeline."""

    TEXT_ENCODER = "TextEncoder"
    TEXT_ENCODER_2 = "TextEncoder2"
    TRANSFORMER = "Transformer"
    VAE = "Vae"


_SUBFOLDER = {
    ModelVariant.TEXT_ENCODER: "text_encoder",
    ModelVariant.TEXT_ENCODER_2: "text_encoder_2",
    ModelVariant.TRANSFORMER: "transformer",
    ModelVariant.VAE: "vae",
}


# ---------------------------------------------------------------------------
# Component loaders — pull a single subfolder, never the whole pipeline.
# ---------------------------------------------------------------------------
def load_text_encoder(dtype: torch.dtype = DTYPE):
    """Qwen2.5-VL text stack from the text_encoder subfolder.

    The pipeline feeds only input_ids/attention_mask (no pixel_values), so the
    vision tower never runs. Return ``.language_model`` (Qwen2_5_VLTextModel,
    ~7.07B) to drop the unused vision tower + LM head that would otherwise be
    uploaded/replicated on every chip.
    """
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(
        REPO_ID, subfolder="text_encoder", torch_dtype=dtype, device_map="cpu"
    ).eval()
    return getattr(encoder, "language_model", encoder)


def load_text_encoder_2(dtype: torch.dtype = DTYPE):
    """ByT5 glyph encoder from the text_encoder_2 subfolder."""
    from transformers import T5EncoderModel

    return T5EncoderModel.from_pretrained(
        REPO_ID, subfolder="text_encoder_2", torch_dtype=dtype, device_map="cpu"
    ).eval()


def load_transformer(dtype: torch.dtype = DTYPE):
    """HunyuanImageTransformer2DModel (MM-DiT) from the transformer subfolder."""
    from diffusers import HunyuanImageTransformer2DModel

    return HunyuanImageTransformer2DModel.from_pretrained(
        REPO_ID, subfolder="transformer", torch_dtype=dtype, device_map="cpu"
    ).eval()


def load_vae(dtype: torch.dtype = DTYPE):
    """AutoencoderKLHunyuanImage from the vae subfolder."""
    from diffusers import AutoencoderKLHunyuanImage

    return AutoencoderKLHunyuanImage.from_pretrained(
        REPO_ID, subfolder="vae", torch_dtype=dtype, device_map="cpu"
    ).eval()


# ---------------------------------------------------------------------------
# Wrappers — tensor-in / tensor-out forward for the graph runner.
# ---------------------------------------------------------------------------
class TextEncoderWrapper(torch.nn.Module):
    """Qwen2.5-VL text stack -> last_hidden_state tensor."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state


class TextEncoder2Wrapper(torch.nn.Module):
    """ByT5 encoder -> last_hidden_state tensor."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state


class HunyuanImage21TransformerWrapper(torch.nn.Module):
    """MM-DiT denoise step -> predicted latent tensor."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states,
        timestep,
        timestep_r,
        guidance,
        encoder_hidden_states,
        encoder_attention_mask,
        encoder_hidden_states_2,
        encoder_attention_mask_2,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep_r,
            guidance=guidance,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states_2=encoder_hidden_states_2,
            encoder_attention_mask_2=encoder_attention_mask_2,
            return_dict=False,
        )[0]


class VAEDecoderWrapper(torch.nn.Module):
    """AutoencoderKLHunyuanImage decoder: latent z -> reconstructed sample."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


# ---------------------------------------------------------------------------
# SPMD shard specifications (Megatron-style column/row parallel).
# ---------------------------------------------------------------------------
def shard_text_encoder_specs(encoder) -> dict:
    """tensor -> partition_spec for the Qwen2.5-VL text stack.

    Column-parallel (q, k, v, gate, up): ("model", "batch")
    Row-parallel    (o, down):           ("batch", "model")
    """
    import os

    # DIAGNOSTIC (HUNYUAN_TE_MLP_ONLY=1): replicate the attention projections and
    # shard only the MLP. Localizes whether the PCC=0.277 failure originates in the
    # sharded attention path (M-RoPE cos/sin + GQA repeat_kv under SPMD) vs the MLP.
    mlp_only = os.environ.get("HUNYUAN_TE_MLP_ONLY") == "1"

    specs = {}
    encoder = getattr(encoder, "language_model", encoder)

    if hasattr(encoder, "embed_tokens"):
        specs[encoder.embed_tokens.weight] = (None, "batch")

    layers = getattr(encoder, "layers", None)
    if not layers:
        raise ValueError(
            f"No decoder layers on {type(encoder).__name__}; refusing to run "
            "fully replicated (expected `.layers` after unwrapping)."
        )

    for layer in layers:
        sa = layer.self_attn
        if not mlp_only:
            for proj in ("q_proj", "k_proj", "v_proj"):
                p = getattr(sa, proj)
                specs[p.weight] = ("model", "batch")
                if p.bias is not None:
                    specs[p.bias] = ("model",)
            specs[sa.o_proj.weight] = ("batch", "model")

        mlp = layer.mlp
        specs[mlp.gate_proj.weight] = ("model", "batch")
        specs[mlp.up_proj.weight] = ("model", "batch")
        specs[mlp.down_proj.weight] = ("batch", "model")

        specs[layer.input_layernorm.weight] = ("batch",)
        specs[layer.post_attention_layernorm.weight] = ("batch",)

    if hasattr(encoder, "norm"):
        specs[encoder.norm.weight] = ("batch",)

    return specs


def shard_transformer_specs(transformer) -> dict:
    """tensor -> partition_spec for HunyuanImageTransformer2DModel.

    Hybrid MM-DiT: ``transformer_blocks`` (dual-stream) +
    ``single_transformer_blocks`` (single-stream).
    Column-parallel (Q,K,V,FFN up): ("model", "batch")
    Row-parallel    (O,FFN down):   ("batch", "model")
    """
    specs = {}

    if hasattr(transformer, "x_embedder") and hasattr(transformer.x_embedder, "proj"):
        specs[transformer.x_embedder.proj.weight] = ("batch", None, None, None)
        if transformer.x_embedder.proj.bias is not None:
            specs[transformer.x_embedder.proj.bias] = ("batch",)

    def _shard_attn(attn):
        for proj_name in (
            "to_q",
            "to_k",
            "to_v",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
        ):
            proj = getattr(attn, proj_name, None)
            if proj is not None:
                specs[proj.weight] = ("model", "batch")
                if proj.bias is not None:
                    specs[proj.bias] = ("model",)
        for proj_name in ("to_out", "to_add_out"):
            out = getattr(attn, proj_name, None)
            if out is not None:
                target = (
                    out[0]
                    if isinstance(out, (torch.nn.Sequential, torch.nn.ModuleList))
                    else out
                )
                specs[target.weight] = ("batch", "model")
                if target.bias is not None:
                    specs[target.bias] = ("batch",)

    def _shard_ff(ff):
        # diffusers FeedForward = Sequential(GEGLU, Dropout, Linear)
        if hasattr(ff, "net"):
            if hasattr(ff.net[0], "proj"):
                specs[ff.net[0].proj.weight] = ("model", "batch")
                if ff.net[0].proj.bias is not None:
                    specs[ff.net[0].proj.bias] = ("model",)
            specs[ff.net[2].weight] = ("batch", "model")
            if ff.net[2].bias is not None:
                specs[ff.net[2].bias] = ("batch",)

    for block in getattr(transformer, "transformer_blocks", []):
        for norm_name in ("norm1", "norm1_context"):
            norm = getattr(block, norm_name, None)
            if norm is not None and hasattr(norm, "linear"):
                specs[norm.linear.weight] = ("model", "batch")
                if norm.linear.bias is not None:
                    specs[norm.linear.bias] = ("model",)
        if hasattr(block, "attn"):
            _shard_attn(block.attn)
        for ff_name in ("ff", "ff_context"):
            if hasattr(block, ff_name):
                _shard_ff(getattr(block, ff_name))

    for block in getattr(transformer, "single_transformer_blocks", []):
        if hasattr(block, "norm") and hasattr(block.norm, "linear"):
            specs[block.norm.linear.weight] = ("model", "batch")
            if block.norm.linear.bias is not None:
                specs[block.norm.linear.bias] = ("model",)
        if hasattr(block, "attn"):
            _shard_attn(block.attn)
        if hasattr(block, "proj_out"):
            specs[block.proj_out.weight] = ("batch", "model")
            if block.proj_out.bias is not None:
                specs[block.proj_out.bias] = ("batch",)

    if hasattr(transformer, "proj_out"):
        specs[transformer.proj_out.weight] = (None, "batch")
        if transformer.proj_out.bias is not None:
            specs[transformer.proj_out.bias] = (None,)

    return specs


class ModelLoader(ForgeModel):
    """Load individual HunyuanImage 2.1 components without the full pipeline."""

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.TEXT_ENCODER_2: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=REPO_ID),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    _TEXT_VARIANTS = (ModelVariant.TEXT_ENCODER, ModelVariant.TEXT_ENCODER_2)

    def __init__(self, variant=None):
        super().__init__(variant)
        self.component_name = _SUBFOLDER[self._variant]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant in cls._TEXT_VARIANTS
            else ModelTask.MM_IMAGE_TTT
        )
        return ModelInfo(
            model="HunyuanImage21",
            variant=variant,
            group=ModelGroup.RED,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return the wrapped component (tensor-in / tensor-out) for this variant."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            return TextEncoderWrapper(load_text_encoder(dtype)).eval()
        if self._variant == ModelVariant.TEXT_ENCODER_2:
            return TextEncoder2Wrapper(load_text_encoder_2(dtype)).eval()
        if self._variant == ModelVariant.TRANSFORMER:
            return HunyuanImage21TransformerWrapper(load_transformer(dtype)).eval()
        if self._variant == ModelVariant.VAE:
            return VAEDecoderWrapper(load_vae(dtype)).eval()

        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Synthetic inputs matching the captured shapes/dtypes for this component."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TEXT_ENCODER:
            input_ids = torch.randint(
                0, QWEN_VOCAB_SIZE, (1, TEXT_TOKEN_MAX_LEN), dtype=torch.long
            )
            attention_mask = torch.ones(1, TEXT_TOKEN_MAX_LEN, dtype=torch.long)
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.TEXT_ENCODER_2:
            input_ids = torch.randint(
                0, BYT5_VOCAB_SIZE, (1, TEXT_TOKEN_2_MAX_LEN), dtype=torch.long
            )
            # Pipeline calls text_encoder_2 with attention_mask.float().
            attention_mask = torch.ones(1, TEXT_TOKEN_2_MAX_LEN, dtype=torch.float32)
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.TRANSFORMER:
            hidden_states = torch.randn(
                1, NUM_CHANNELS_LATENTS, LATENT_H, LATENT_W, dtype=dtype
            )
            timestep = torch.tensor([1000.0], dtype=dtype)
            timestep_r = torch.tensor([0.0], dtype=dtype)
            guidance = torch.tensor(
                [3500.0], dtype=dtype
            )  # distilled_guidance_scale * 1000
            encoder_hidden_states = torch.randn(
                1, TRANSFORMER_TEXT_SEQ, TEXT_EMBED_DIM, dtype=dtype
            )
            encoder_attention_mask = torch.ones(
                1, TRANSFORMER_TEXT_SEQ, dtype=torch.long
            )
            encoder_hidden_states_2 = torch.randn(
                1, TRANSFORMER_TEXT_2_SEQ, TEXT_EMBED_2_DIM, dtype=dtype
            )
            encoder_attention_mask_2 = torch.ones(
                1, TRANSFORMER_TEXT_2_SEQ, dtype=torch.long
            )
            return [
                hidden_states,
                timestep,
                timestep_r,
                guidance,
                encoder_hidden_states,
                encoder_attention_mask,
                encoder_hidden_states_2,
                encoder_attention_mask_2,
            ]

        if self._variant == ModelVariant.VAE:
            z = torch.randn(1, NUM_CHANNELS_LATENTS, LATENT_H, LATENT_W, dtype=dtype)
            return [z]

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        Single-device-only components (TEXT_ENCODER_2, VAE) collapse to (1, 1).
        """
        if self._variant in (ModelVariant.TEXT_ENCODER_2, ModelVariant.VAE):
            return (1, 1), MESH_NAMES
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return tensor -> partition_spec for the active (wrapped) component.

        ``model`` is the wrapper returned by load_model(); reach into the inner
        module. TEXT_ENCODER_2 and VAE are single-device (no sharding).
        """
        if self._variant == ModelVariant.TEXT_ENCODER:
            return shard_text_encoder_specs(model.encoder)
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_transformer_specs(model.transformer)
        return None
