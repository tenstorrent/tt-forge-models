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
encoder do not fit a single n150 (7 B/chip) device; the VAE does. Both heavy
components are brought up tensor-parallel on an 8-chip n300 llmbox --
``get_mesh_config`` gives the ("batch", "model") mesh and ``load_shard_spec``
the per-parameter partition specs. The VAE stays single-device (replicated).
"""

import types
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

# ---- multichip SPMD tensor-parallel mesh (FSDP-style ("batch", "model")) -----
# Both heavy components are weight-bound on a single chip and use a 2D
# ("batch", "model") mesh (Megatron column/row weights spread over both axes).
# The "model" axis is the tensor-parallel degree; it must divide both:
#   transformer  : num_attention_heads = 24        -> 24 % model == 0
#   text_encoder : num_attention_heads = 28,        -> 28 % model == 0 AND
#                  num_key_value_heads = 4 (GQA)        model <= 4 (KV-head cap)
# The intersection caps the model axis at 4, so on an 8-chip n300 llmbox the
# mesh is (2, 4). The VAE fits a single chip and simply replicates on the mesh.
MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
MESH_NAMES = ("batch", "model")

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


def _single_block_forward_split(
    self,
    hidden_states,
    encoder_hidden_states,
    temb,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
):
    """TP-friendly rewrite of LongCatImageSingleTransformerBlock.forward.

    Mathematically identical to the stock forward, but replaces the fused
    ``proj_out(cat([attn_output, mlp_hidden_states], dim=-1))`` with two
    separate linears summed:

        proj_out(cat([a, m])) == proj_out_attn(a) + proj_out_mlp(m)

    where ``proj_out.weight`` [out, 3072+12288] is column-split into the attn
    half [:, :3072] and the mlp half [:, 3072:]. This lets each half be a clean
    row-parallel (None, "model") matmul aligned to its column-sharded operand
    (attn heads / proj_mlp), so the single block can be tensor-parallel without
    the block-interleaved concat that forced a gather + wide L1 overflow.
    """
    text_seq_len = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    joint_attention_kwargs = joint_attention_kwargs or {}
    attn_output = self.attn(
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )

    proj = self.proj_out_attn(attn_output) + self.proj_out_mlp(mlp_hidden_states)
    gate = gate.unsqueeze(1)
    hidden_states = gate * proj
    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    encoder_hidden_states, hidden_states = (
        hidden_states[:, :text_seq_len],
        hidden_states[:, text_seq_len:],
    )
    return encoder_hidden_states, hidden_states


def _split_single_block_proj_out(block) -> None:
    """Split a single block's fused ``proj_out`` into attn + mlp halves in place.

    The concatenated ``proj_out`` input is [attn_output (attn_dim) | mlp_hidden
    (mlp_dim)], so proj_out.weight columns [:attn_dim] act on the attention
    output and [attn_dim:] on the mlp hidden. We carve them into two Linears
    (bias kept once on the attn half) and rebind the block's forward to the
    concat-free variant. The original fused ``proj_out`` is removed so its
    weight is not uploaded (replicated) to device.
    """
    old = block.proj_out  # Linear(attn_dim + mlp_dim -> out)
    w = old.weight.data  # [out, attn_dim + mlp_dim]
    out_features = old.out_features
    attn_dim = block.attn.to_v.weight.shape[0]  # attention output feature dim
    mlp_dim = w.shape[1] - attn_dim

    attn_lin = torch.nn.Linear(
        attn_dim, out_features, bias=old.bias is not None, dtype=w.dtype
    )
    attn_lin.weight.data.copy_(w[:, :attn_dim])
    if old.bias is not None:
        attn_lin.bias.data.copy_(old.bias.data)

    mlp_lin = torch.nn.Linear(mlp_dim, out_features, bias=False, dtype=w.dtype)
    mlp_lin.weight.data.copy_(w[:, attn_dim:])

    block.proj_out_attn = attn_lin
    block.proj_out_mlp = mlp_lin
    del block.proj_out
    block.forward = types.MethodType(_single_block_forward_split, block)


class _LongCatTransformerWrapper(torch.nn.Module):
    """Adapt LongCatImageTransformer2DModel to a tensors-only forward.

    The structural position-id args (img_ids / txt_ids) are reconstructed at
    the captured 256x256 / 512-token layout and pinned as buffers; guidance is
    None (guidance_embeds is disabled in this checkpoint). The runner therefore
    only feeds hidden_states / timestep / encoder_hidden_states.
    """

    def __init__(self, transformer):
        super().__init__()
        # Rewrite each single block to a concat-free, tensor-parallel-friendly
        # forward (split proj_out into attn + mlp halves) so the single blocks
        # can be sharded instead of fully replicated (~5.7 GB) on every chip.
        for block in getattr(transformer, "single_transformer_blocks", []):
            _split_single_block_proj_out(block)
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


# ---------------------------------------------------------------------------
# SPMD tensor-parallel shard specifications (Megatron 1D on the "model" axis)
# Column-parallel weights (Q/K/V, gate/up, ff in): ("model", None)
# Row-parallel weights    (O, down, ff out):       (None, "model")
# Column bias -> ("model",); row bias / layernorm / stem -> (None,) or replicate.
# The mesh may be 2D ((batch, model)); the "batch" axis stays unused for weights
# (activation data-parallel), matching the flux2 Flux-family reference. 1D specs
# avoid the cross-axis reshards that produce unsupported CollectivePermuteOps.
# ---------------------------------------------------------------------------


def _shard_text_encoder_specs(text_encoder) -> dict:
    """Shard specs for the Qwen2.5-VL text encoder (Megatron 1D column/row).

    ``text_encoder`` is the Qwen2_5_VLForConditionalGeneration held by
    ``_LongCatTextEncoderWrapper``. Only the language model is exercised by the
    forward (no pixel_values), so the vision tower / lm_head stay out of the
    graph and are never uploaded. We navigate to the decoder and shard its
    attention + MLP weights.
    """
    # Qwen2_5_VLForConditionalGeneration -> model -> language_model (decoder).
    decoder = None
    for path in ("model.language_model", "language_model", "model"):
        obj = text_encoder
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and hasattr(obj, "layers"):
            decoder = obj
            break
    if decoder is None:
        raise ValueError(
            f"Could not locate decoder layers on {type(text_encoder).__name__}; "
            "refusing to run fully replicated (would DRAM OOM)."
        )

    specs = {}
    if hasattr(decoder, "embed_tokens"):
        specs[decoder.embed_tokens.weight] = (None, None)

    for layer in decoder.layers:
        sa = layer.self_attn
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            proj = getattr(sa, proj_name)
            specs[proj.weight] = ("model", None)
            if proj.bias is not None:
                specs[proj.bias] = ("model",)
        specs[sa.o_proj.weight] = (None, "model")
        if sa.o_proj.bias is not None:
            specs[sa.o_proj.bias] = (None,)

        mlp = layer.mlp
        specs[mlp.gate_proj.weight] = ("model", None)
        specs[mlp.up_proj.weight] = ("model", None)
        specs[mlp.down_proj.weight] = (None, "model")

        specs[layer.input_layernorm.weight] = (None,)
        specs[layer.post_attention_layernorm.weight] = (None,)

    if hasattr(decoder, "norm"):
        specs[decoder.norm.weight] = (None,)

    return specs


def _shard_attn_specs(attn, specs: dict) -> None:
    """Column-shard Q/K/V (+ added_kv projections) and row-shard the outputs.

    Per-head RMSNorms (norm_q/norm_k/...) shard on the head (model) axis; they
    act per-head so they follow the Q/K sharding.
    """
    for proj_name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
        proj = getattr(attn, proj_name, None)
        if proj is not None:
            specs[proj.weight] = ("model", None)
            if proj.bias is not None:
                specs[proj.bias] = ("model",)
    # to_out is an nn.ModuleList([Linear, Dropout]); to_add_out is a plain Linear.
    to_out = getattr(attn, "to_out", None)
    if to_out is not None:
        out_lin = to_out[0] if isinstance(to_out, (torch.nn.ModuleList,)) else to_out
        specs[out_lin.weight] = (None, "model")
        if out_lin.bias is not None:
            specs[out_lin.bias] = (None,)
    to_add_out = getattr(attn, "to_add_out", None)
    if to_add_out is not None:
        specs[to_add_out.weight] = (None, "model")
        if to_add_out.bias is not None:
            specs[to_add_out.bias] = (None,)


def _shard_ff_specs(ff, specs: dict) -> None:
    """diffusers FeedForward = Sequential(GELU(proj), Dropout, Linear)."""
    net = ff.net
    if hasattr(net[0], "proj"):
        specs[net[0].proj.weight] = ("model", None)
        if net[0].proj.bias is not None:
            specs[net[0].proj.bias] = ("model",)
    specs[net[2].weight] = (None, "model")
    if net[2].bias is not None:
        specs[net[2].bias] = (None,)


def _shard_modulation_linear(linear, specs: dict) -> None:
    """Row-parallel a shared-input AdaLayerNorm modulation linear.

    All AdaLayerNorm modulation linears (dual norm1/norm1_context, single norm,
    final norm_out) take the SAME timestep embedding as input, so the compiler
    const-evals them into one fused matmul by concatenating every (transposed)
    weight into a single [3072, sum(out)] buffer -- ~6.87 GB in f32 across all
    41 linears, materialized on ONE device -> DRAM OOM.

    Row-parallel (None, "model") shards each weight on its shared in dim (3072),
    so that fused buffer is sharded ~1/model per chip. Crucially, the matmul
    output is all-reduced back to a REPLICATED activation, so the downstream
    chunk() into modulation slices never lands on a sharded dim -- avoiding the
    CollectivePermute that column-sharding these linears would trigger
    (https://github.com/tenstorrent/tt-mlir/issues/3370). Bias is added after
    the reduction and stays replicated.
    """
    if linear is None:
        return
    specs[linear.weight] = (None, "model")
    if getattr(linear, "bias", None) is not None:
        specs[linear.bias] = (None,)


def _shard_single_block_specs(block, specs: dict) -> None:
    """Shard one LongCatImageSingleTransformerBlock (Megatron column/row).

    The block was rewritten by ``_split_single_block_proj_out`` into a
    concat-free forward with ``proj_out_attn`` / ``proj_out_mlp``. Column-shard
    the attention Q/K/V and ``proj_mlp``; row-shard both proj_out halves so each
    matches its column-sharded operand. The AdaLayerNormZeroSingle modulation
    linear (``norm.linear``) is handled by the caller via
    ``_shard_modulation_linear`` (row-parallel, see NOTE 1).
    """
    attn = getattr(block, "attn", None)
    if attn is not None:
        for proj_name in ("to_q", "to_k", "to_v"):
            proj = getattr(attn, proj_name, None)
            if proj is not None:
                specs[proj.weight] = ("model", None)
                if proj.bias is not None:
                    specs[proj.bias] = ("model",)

    proj_mlp = getattr(block, "proj_mlp", None)
    if proj_mlp is not None:
        specs[proj_mlp.weight] = ("model", None)
        if proj_mlp.bias is not None:
            specs[proj_mlp.bias] = ("model",)

    # proj_out split halves: row-parallel, matched to their column-sharded
    # operands (attn heads / proj_mlp). Bias (kept on the attn half) replicates.
    for out_name in ("proj_out_attn", "proj_out_mlp"):
        out_lin = getattr(block, out_name, None)
        if out_lin is not None:
            specs[out_lin.weight] = (None, "model")
            if out_lin.bias is not None:
                specs[out_lin.bias] = (None,)


def _shard_transformer_specs(transformer) -> dict:
    """Shard specs for LongCatImageTransformer2DModel (Flux-style MMDiT).

    10 dual-stream ``transformer_blocks`` (image + text streams) and 20
    ``single_transformer_blocks``. Both block types shard their attention +
    feed-forward matmuls; only the AdaLayerNorm modulation linears replicate.
    See the design note below.
    """
    specs = {}

    # NOTE 1 — AdaLayerNorm modulation linears (dual norm1/norm1_context, single
    # norm, final norm_out) are ROW-parallel (see _shard_modulation_linear).
    # Column-sharding them would make the chunk() modulation slice a sharded dim
    # -> an unsupported CollectivePermuteOp (tt-mlir #3370); leaving them fully
    # REPLICATED instead makes the compiler concat all 41 into one ~6.87 GB f32
    # buffer on a single device -> DRAM OOM. Row-parallel shards the shared in
    # dim (so the fused buffer shrinks ~1/model) while the all-reduced output
    # stays replicated for the chunk().
    #
    # NOTE 2 — single-stream blocks were previously left FULLY REPLICATED (the
    # bulk of the weight, ~5.7 GB) because LongCat's single block runs an
    # explicit torch.cat([attn_out, mlp_hidden], dim=-1) before proj_out, whose
    # block-interleaved layout a contiguous row-parallel proj_out could not
    # match. ``_split_single_block_proj_out`` rewrites that block to sum two
    # column-split proj_out halves instead of concatenating, so the single
    # blocks are now sharded like the dual blocks.

    # Dual-stream blocks: shard attention + both feed-forwards + modulation.
    for block in getattr(transformer, "transformer_blocks", []):
        if hasattr(block, "attn"):
            _shard_attn_specs(block.attn, specs)
        for ff_name in ("ff", "ff_context"):
            ff = getattr(block, ff_name, None)
            if ff is not None:
                _shard_ff_specs(ff, specs)
        for norm_name in ("norm1", "norm1_context"):
            norm = getattr(block, norm_name, None)
            if norm is not None and hasattr(norm, "linear"):
                _shard_modulation_linear(norm.linear, specs)

    # Single-stream blocks: shard attention + proj_mlp + split proj_out halves
    # + modulation.
    for block in getattr(transformer, "single_transformer_blocks", []):
        _shard_single_block_specs(block, specs)
        norm = getattr(block, "norm", None)
        if norm is not None and hasattr(norm, "linear"):
            _shard_modulation_linear(norm.linear, specs)

    # Final AdaLayerNormContinuous modulation linear (shares the same input).
    norm_out = getattr(transformer, "norm_out", None)
    if norm_out is not None and hasattr(norm_out, "linear"):
        _shard_modulation_linear(norm_out.linear, specs)

    return specs


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

            vae = AutoencoderKL.from_pretrained(repo, **_component_kwargs(dtype, "vae"))
            return _LongCatVAEDecoderWrapper(vae.eval())

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        Every component attaches to the same mesh for fabric uniformity; the
        VAE replicates (load_shard_spec -> None). The "model" axis is capped at
        4 by the text encoder's GQA (4 KV heads), so 8 chips map to (2, 4).
        """
        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES

    def load_shard_spec(self, model):
        """Return {param -> partition_spec} for the active component.

        Expects the module returned by load_model():
          TEXT_ENCODER -> _LongCatTextEncoderWrapper (specs from .text_encoder)
          TRANSFORMER  -> _LongCatTransformerWrapper (specs from .transformer)
          VAE          -> None (fits a single chip; replicate on the mesh)
        """
        if self._variant == ModelVariant.TEXT_ENCODER:
            return _shard_text_encoder_specs(model.text_encoder)
        if self._variant == ModelVariant.TRANSFORMER:
            return _shard_transformer_specs(model.transformer)
        return None

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
            latent = torch.randn(B, VAE_Z_CHANNELS, VAE_Z_H, VAE_Z_W, dtype=dtype)
            return [latent]

        raise ValueError(f"Unknown variant: {self._variant}")
