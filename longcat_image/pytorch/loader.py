# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LongCat-Image per-component loader.

meituan-longcat/LongCat-Image is a LongCatImagePipeline (bilingual text-to-image
MMDiT, ~6B headline transformer):
  text_encoder  -> Qwen2_5_VLForConditionalGeneration (Qwen2.5-VL 7B)   params ~7.7 B
  transformer   -> LongCatImageTransformer2DModel (Flux-style, 10 dual   params ~6 B
                   + 20 single blocks, joint_attention_dim 3584)
  vae           -> AutoencoderKL (16-channel 2D, 8x spatial compression)  params ~0.08 B
  tokenizer     -> Qwen2Tokenizer                  (no parameters, skipped)
  text_processor-> Qwen2VLProcessor                (no parameters, skipped)
  scheduler     -> FlowMatchEulerDiscreteScheduler (no parameters, skipped)

Each variant scaffolds one component as an independent torch.nn.Module the
runner can compile + PCC-compare in isolation. The full pipeline is never
loaded -- each component is fetched directly via from_pretrained(..., subfolder=...).


NOTE: LongCatImagePipeline requires diffusers >= 0.36 (LongCat classes landed
in diffusers main, Dec 2025; first released in 0.36.0). See requirements.txt.

NOTE: this is a ~14 B aggregate pipeline. The 6 B transformer and 7.7 B text
encoder do not fit a single n150 (7 B/chip) device; the VAE does. Components
are brought up on a single device first (SHARD_SPECS / TT_VISIBLE_DEVICES
record the tensor-parallel plan for the multi-chip follow-up). Scaffolded
CPU-only; per-component single-device results recorded in the bringup state.
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

# Repo
LONGCAT_IMAGE_REPO_ID = "meituan-longcat/LongCat-Image"

DTYPE = torch.bfloat16

# ---- captured I/O spec (256x256, 2 denoise steps, prompt-rewrite off; CPU bf16) ----
_COMPONENT_IO_SPEC = {
    "text_encoder": {
        "class": "Qwen2_5_VLForConditionalGeneration",
        "inputs": {
            "input_ids": {"shape": (1, 553), "dtype": "torch.int64"},
            "attention_mask": {"shape": (1, 553), "dtype": "torch.int64"},
        },
        "output": {
            "last_hidden_state": {"shape": (1, 553, 3584), "dtype": "torch.bfloat16"}
        },
    },
    "transformer": {
        "class": "LongCatImageTransformer2DModel",
        "inputs": {
            "hidden_states": {"shape": (1, 256, 64), "dtype": "torch.bfloat16"},
            "timestep": {"shape": (1,), "dtype": "torch.bfloat16"},
            "encoder_hidden_states": {
                "shape": (1, 512, 3584),
                "dtype": "torch.bfloat16",
            },
            # pinned structural args (reconstructed from prepare_pos_ids):
            "txt_ids": {"shape": (512, 3), "dtype": "torch.float32"},
            "img_ids": {"shape": (256, 3), "dtype": "torch.float32"},
            "guidance": None,
        },
        "output": {"shape": (1, 256, 64), "dtype": "torch.bfloat16"},
    },
    "vae": {
        "class": "AutoencoderKL",
        "inputs": {"latent": {"shape": (1, 16, 32, 32), "dtype": "torch.bfloat16"}},
        "op": "decode",
    },
}

# ---- tensor-parallel shard plan ---------------------------------------------
# The transformer (~6.3 B) and text encoder (~7.7 B) do not fit a single 12 GB
# Wormhole chip (n300 = 2 chips). They are sharded Megatron-style across the
# "model" mesh axis: column-parallel on QKV / FFN-up (out_features split),
# row-parallel on the output / FFN-down projections (in_features split), with an
# all-reduce at each row-parallel output. Norms, modulation (adaLN) linears, and
# the small in/out embedders stay replicated. The VAE (~0.08 B) fits one chip.
#
# Mesh axes are ("batch", "model"); for n300 the active shape is (1, 2).
MESH_NAMES = ("batch", "model")
MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}


def _add_spec(specs: dict, param: Optional[torch.Tensor], spec: tuple) -> None:
    """Register a partition spec only for real (non-None) parameters."""
    if param is not None:
        specs[param] = spec


def _shard_linear_col(specs: dict, lin) -> None:
    """Column-parallel: split out_features across the model axis."""
    _add_spec(specs, lin.weight, ("model", None))
    _add_spec(specs, getattr(lin, "bias", None), ("model",))


def _shard_linear_row(specs: dict, lin) -> None:
    """Row-parallel: split in_features; output is all-reduced (bias replicated)."""
    _add_spec(specs, lin.weight, (None, "model"))
    _add_spec(specs, getattr(lin, "bias", None), (None,))


def _replicate_norm(specs: dict, norm) -> None:
    if norm is not None and getattr(norm, "weight", None) is not None:
        _add_spec(specs, norm.weight, (None,))
        _add_spec(specs, getattr(norm, "bias", None), (None,))


def _shard_longcat_attention(attn, specs: dict) -> None:
    """LongCatImageAttention: joint image/text MMDiT attention.

    Heads are split across the model axis: image QKV (to_q/k/v) and text QKV
    (add_*_proj) are column-parallel; the output projections (to_out[0],
    to_add_out) are row-parallel. Per-head RMSNorms are replicated. Single-stream
    blocks reuse the same attention module but omit to_out / the text projections
    (their output projection lives at block level in proj_out).
    """
    for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
        proj = getattr(attn, name, None)
        if proj is not None:
            _shard_linear_col(specs, proj)

    to_out = getattr(attn, "to_out", None)
    if to_out is not None:
        out_proj = (
            to_out[0]
            if isinstance(to_out, (torch.nn.ModuleList, torch.nn.Sequential))
            else to_out
        )
        _shard_linear_row(specs, out_proj)

    if getattr(attn, "to_add_out", None) is not None:
        _shard_linear_row(specs, attn.to_add_out)

    for norm_name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        _replicate_norm(specs, getattr(attn, norm_name, None))


def shard_transformer_specs(transformer) -> dict:
    """Megatron-style tensor-parallel specs for LongCatImageTransformer2DModel."""
    specs: dict = {}

    # In/out embedders and final projection: small, replicated.
    _add_spec(specs, transformer.x_embedder.weight, (None, None))
    _add_spec(specs, getattr(transformer.x_embedder, "bias", None), (None,))
    _add_spec(specs, transformer.context_embedder.weight, (None, None))
    _add_spec(specs, getattr(transformer.context_embedder, "bias", None), (None,))
    _add_spec(specs, transformer.proj_out.weight, (None, None))
    _add_spec(specs, getattr(transformer.proj_out, "bias", None), (None,))

    # Dual-stream blocks.
    for block in transformer.transformer_blocks:
        # adaLN modulation linears (norm1 / norm1_context) -> replicated.
        _add_spec(specs, block.norm1.linear.weight, (None, None))
        _add_spec(specs, getattr(block.norm1.linear, "bias", None), (None,))
        _add_spec(specs, block.norm1_context.linear.weight, (None, None))
        _add_spec(specs, getattr(block.norm1_context.linear, "bias", None), (None,))

        _shard_longcat_attention(block.attn, specs)

        # FeedForward: net[0] is GELU(proj=up Linear), net[2] is down Linear.
        for ff in (block.ff, block.ff_context):
            _shard_linear_col(specs, ff.net[0].proj)
            _shard_linear_row(specs, ff.net[2])

    # Single-stream blocks (fused attention + MLP).
    for block in transformer.single_transformer_blocks:
        _add_spec(specs, block.norm.linear.weight, (None, None))
        _add_spec(specs, getattr(block.norm.linear, "bias", None), (None,))
        _shard_longcat_attention(block.attn, specs)
        # MLP up-projection (column) and the fused [attn|mlp] -> dim out (row).
        _shard_linear_col(specs, block.proj_mlp)
        _shard_linear_row(specs, block.proj_out)

    return specs


def shard_text_encoder_specs(text_encoder) -> dict:
    """Megatron-style specs for the Qwen2.5-VL language-model decoder layers.

    The vision tower is unused for text-only prompt encoding and stays replicated.
    """
    specs: dict = {}
    lm = text_encoder
    for attr in ("model", "language_model"):
        if hasattr(lm, attr):
            lm = getattr(lm, attr)

    layers = getattr(lm, "layers", None)
    if layers is None:
        return specs

    for layer in layers:
        sa = layer.self_attn
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            _shard_linear_col(specs, getattr(sa, proj_name))
        _shard_linear_row(specs, sa.o_proj)

        mlp = layer.mlp
        _shard_linear_col(specs, mlp.gate_proj)
        _shard_linear_col(specs, mlp.up_proj)
        _shard_linear_row(specs, mlp.down_proj)

        _replicate_norm(specs, getattr(layer, "input_layernorm", None))
        _replicate_norm(specs, getattr(layer, "post_attention_layernorm", None))

    _replicate_norm(specs, getattr(lm, "norm", None))
    return specs

# ---- shape constants (from captured spec) ----------------------------------
TE_SEQ_LEN = 553
TE_HIDDEN = 3584
TE_VOCAB = 152064

TR_LATENT_SEQ = 256  # (256/8/2)^2 = 16*16 packed patch tokens for 256x256
TR_IN_CHANNELS = 64
TR_TXT_SEQ = 512  # == tokenizer_max_length
TR_JOINT_DIM = 3584
TR_LATENT_PATCH_HW = 16  # latent h//2 == w//2 for 256x256
TOKENIZER_MAX_LENGTH = 512  # image-id position offset (pipeline.tokenizer_max_length)

VAE_Z_CHANNELS = 16
VAE_Z_H = 32  # 256 / vae_scale_factor(8)
VAE_Z_W = 32


def _prepare_pos_ids(
    modality_id=0, type="text", start=(0, 0), num_token=None, height=None, width=None
):
    """Reconstruct LongCatImagePipeline.prepare_pos_ids (verified bit-exact
    against the captured txt_ids/img_ids). Position ids are structural -- they
    depend only on the resolution / sequence length, not on the data."""
    if type == "text":
        pos_ids = torch.zeros(num_token, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = torch.arange(num_token) + start[0]
        pos_ids[..., 2] = torch.arange(num_token) + start[1]
    elif type == "image":
        pos_ids = torch.zeros(height, width, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = pos_ids[..., 1] + torch.arange(height)[:, None] + start[0]
        pos_ids[..., 2] = pos_ids[..., 2] + torch.arange(width)[None, :] + start[1]
        pos_ids = pos_ids.reshape(height * width, 3)
    else:
        raise KeyError(f'Unknown type {type}, only "text" or "image".')
    return pos_ids


def _component_kwargs(dtype: torch.dtype, subfolder: str) -> dict:
    return {"subfolder": subfolder, "torch_dtype": dtype}


class _LongCatTextEncoderWrapper(torch.nn.Module):
    """Adapt the Qwen2.5-VL text encoder to forward(input_ids, attention_mask)
    -> last hidden state [B, seq, 3584] (the prompt embedding the pipeline uses)."""

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids, attention_mask):
        out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return out.hidden_states[-1]


class _LongCatTransformerWrapper(torch.nn.Module):
    """Adapt LongCatImageTransformer2DModel to a tensors-only forward.

    The structural position-id args (img_ids / txt_ids) are reconstructed at
    the captured 256x256 / 512-token layout and pinned as buffers; guidance is
    None (guidance_embeds is disabled in this checkpoint). The runner therefore
    only feeds hidden_states / timestep / encoder_hidden_states.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        txt_ids = _prepare_pos_ids(
            modality_id=0, type="text", start=(0, 0), num_token=TR_TXT_SEQ
        )
        img_ids = _prepare_pos_ids(
            modality_id=1,
            type="image",
            start=(TOKENIZER_MAX_LENGTH, TOKENIZER_MAX_LENGTH),
            height=TR_LATENT_PATCH_HW,
            width=TR_LATENT_PATCH_HW,
        )
        # non-persistent: structural, never part of the checkpoint
        self.register_buffer("txt_ids", txt_ids, persistent=False)
        self.register_buffer("img_ids", img_ids, persistent=False)

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        out = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=self.img_ids,
            txt_ids=self.txt_ids,
            guidance=None,
            return_dict=False,
        )
        return out[0]


class _LongCatVAEDecoderWrapper(torch.nn.Module):
    """Expose AutoencoderKL.decode as forward(latent) -> image."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent, return_dict=False)[0]


class ModelVariant(StrEnum):
    """Loadable components of the LongCat-Image pipeline."""

    TEXT_ENCODER = "LongCat_Image_TextEncoder"
    TRANSFORMER = "LongCat_Image_Transformer"
    VAE = "LongCat_Image_Vae"


class ModelLoader(ForgeModel):
    """Per-component loader for LongCat-Image.

    load_model() returns just the requested component (wrapped to a clean
    tensors-only forward). load_inputs() builds synthetic tensors at the
    captured shapes. The full pipeline is never loaded.
    """

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(
            pretrained_model_name=LONGCAT_IMAGE_REPO_ID
        ),
        ModelVariant.TRANSFORMER: ModelConfig(
            pretrained_model_name=LONGCAT_IMAGE_REPO_ID
        ),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=LONGCAT_IMAGE_REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

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
            model="LongCatImage",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        The VAE fits a single chip, so it maps to (1, 1) regardless of device
        count. The transformer and text encoder shard across the model axis.
        """
        if self._variant == ModelVariant.VAE:
            return (1, 1), MESH_NAMES
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return {parameter -> partition_spec} for tensor-parallel execution.

        ``model`` is the wrapper returned by ``load_model``; the underlying
        component is reached via ``.transformer`` / ``.text_encoder``.
        Unlisted parameters default to replicated.
        """
        if self._variant == ModelVariant.TRANSFORMER:
            return shard_transformer_specs(model.transformer)
        if self._variant == ModelVariant.TEXT_ENCODER:
            return shard_text_encoder_specs(model.text_encoder)
        return None

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        dtype = dtype_override if dtype_override is not None else DTYPE
        repo = self._variant_config.pretrained_model_name

        if self._variant == ModelVariant.TEXT_ENCODER:
            from transformers import Qwen2_5_VLForConditionalGeneration

            te = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                repo, **_component_kwargs(dtype, "text_encoder")
            )
            return _LongCatTextEncoderWrapper(te.eval())

        if self._variant == ModelVariant.TRANSFORMER:
            from diffusers import LongCatImageTransformer2DModel

            transformer = LongCatImageTransformer2DModel.from_pretrained(
                repo, **_component_kwargs(dtype, "transformer")
            )
            return _LongCatTransformerWrapper(transformer.eval())

        if self._variant == ModelVariant.VAE:
            from diffusers import AutoencoderKL

            # This AutoencoderKL ships force_upcast=True: diffusers always runs it
            # in float32 even inside a bf16 pipeline. bf16 conv2d also has no
            # optimized CPU kernel (a single decode does not finish in minutes, so
            # the runner's CPU golden hangs), so pin the VAE to float32 on both the
            # CPU reference and the device. dtype_override is intentionally ignored.
            vae = AutoencoderKL.from_pretrained(
                repo, **_component_kwargs(torch.float32, "vae")
            )
            return _LongCatVAEDecoderWrapper(vae.eval())

        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(
        self,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        **kwargs,
    ):
        """Return synthetic inputs matching each component's forward signature."""
        dtype = dtype_override if dtype_override is not None else DTYPE
        B = batch_size

        if self._variant == ModelVariant.TEXT_ENCODER:
            input_ids = torch.randint(0, TE_VOCAB, (B, TE_SEQ_LEN), dtype=torch.long)
            attention_mask = torch.ones(B, TE_SEQ_LEN, dtype=torch.long)
            return [input_ids, attention_mask]

        if self._variant == ModelVariant.TRANSFORMER:
            hidden_states = torch.randn(B, TR_LATENT_SEQ, TR_IN_CHANNELS, dtype=dtype)
            timestep = torch.ones(B, dtype=dtype)
            encoder_hidden_states = torch.randn(
                B, TR_TXT_SEQ, TR_JOINT_DIM, dtype=dtype
            )
            return [hidden_states, timestep, encoder_hidden_states]

        if self._variant == ModelVariant.VAE:
            # VAE is pinned to float32 (force_upcast=True; bf16 conv2d is
            # unusably slow on CPU). Match the model dtype regardless of override.
            latent = torch.randn(
                B, VAE_Z_CHANNELS, VAE_Z_H, VAE_Z_W, dtype=torch.float32
            )
            return [latent]

        raise ValueError(f"Unknown variant: {self._variant}")
