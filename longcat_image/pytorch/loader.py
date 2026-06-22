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

NOTE: this is a ~14 B aggregate pipeline brought up at the model's native
1024x1024 resolution (pipeline default_sample_size 128 * vae_scale_factor 8).
At that resolution the packed-latent sequence is (1024/16)^2 = 4096 tokens
(== scheduler max_image_seq_len) and the text-conditioning sequence is the
tokenizer_max_length of 512. The 6 B transformer and 7.7 B text encoder each
exceed a single Wormhole chip's 12 GB DRAM once activations are added, so on
n300 (2x Wormhole, 24 GB) they are sharded tensor-parallel across both chips;
the tiny VAE fits a single chip. SHARD_SPECS / TT_VISIBLE_DEVICES record that
plan. The full pipeline is never traced as one graph -- the scheduler /
denoising loop stays in host Python (Step-7 composite-component approach).
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

# ---- I/O spec at native 1024x1024 (prompt-rewrite off; CPU bf16) ------------
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
            "hidden_states": {"shape": (1, 4096, 64), "dtype": "torch.bfloat16"},
            "timestep": {"shape": (1,), "dtype": "torch.bfloat16"},
            "encoder_hidden_states": {
                "shape": (1, 512, 3584),
                "dtype": "torch.bfloat16",
            },
            # pinned structural args (reconstructed from prepare_pos_ids):
            "txt_ids": {"shape": (512, 3), "dtype": "torch.float32"},
            "img_ids": {"shape": (4096, 3), "dtype": "torch.float32"},
            "guidance": None,
        },
        "output": {"shape": (1, 4096, 64), "dtype": "torch.bfloat16"},
    },
    "vae": {
        "class": "AutoencoderKL",
        "inputs": {"latent": {"shape": (1, 16, 128, 128), "dtype": "torch.bfloat16"}},
        "op": "decode",
    },
}

# ---- shard plan (n300 = 2x Wormhole, 12 GB/chip) ----------------------------
# At native 1024x1024 (4096-token latent seq), the 6 B transformer and 7.7 B
# text encoder each exceed one chip's 12 GB DRAM once activations/KV are added,
# so both are tensor-parallel across the 2 chips; the ~0.08 B VAE stays on one.
TT_VISIBLE_DEVICES = "0,1"  # n300 -> 2 Wormhole chips
SHARD_SPECS = {
    "text_encoder": {"strategy": "tensor_parallel", "mesh": [1, 2]},
    "transformer": {"strategy": "tensor_parallel", "mesh": [1, 2]},
    "vae": {"strategy": "data_parallel"},
}

# ---- shape constants (native 1024x1024) ------------------------------------
TE_SEQ_LEN = 553  # prefix + tokenizer_max_length(512) + suffix; resolution-independent
TE_HIDDEN = 3584
TE_VOCAB = 152064

TR_LATENT_SEQ = 4096  # (1024/16)^2 == 64*64 packed patch tokens for 1024x1024
TR_IN_CHANNELS = 64
TR_TXT_SEQ = 512  # == tokenizer_max_length
TR_JOINT_DIM = 3584
TR_LATENT_PATCH_HW = 64  # latent h//2 == w//2 for 1024x1024
TOKENIZER_MAX_LENGTH = 512  # image-id position offset (pipeline.tokenizer_max_length)

VAE_Z_CHANNELS = 16
VAE_Z_H = 128  # 1024 / vae_scale_factor(8)
VAE_Z_W = 128


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


def _shard_transformer_specs(transformer) -> dict:
    """Megatron tensor-parallel spec for LongCatImageTransformer2DModel.

    Flux-style MMDiT (hidden 3072 = 24 heads x 128). Within each block:
      Q/K/V (and the text-stream add_*_proj): column-parallel  ("model", None),
        bias ("model",) -- shards the head dimension across chips.
      output projections (to_out.0, to_add_out): row-parallel (None, "model"),
        bias replicated (None,) -- the post-matmul all-reduce yields the full
        output on every chip, so the bias is added once, replicated.
      FeedForward net.0.proj: column-parallel; net.2: row-parallel.
    Single blocks fuse attention + MLP: to_q/k/v and proj_mlp are column, and
    proj_out (Linear over the concatenated [attn|mlp] feature) is row-parallel.
    AdaLN modulation (norm*.linear), embedders and proj_out stay replicated.
    """
    specs = {}

    def col(linear):
        specs[linear.weight] = ("model", None)
        if linear.bias is not None:
            specs[linear.bias] = ("model",)

    def row(linear):
        specs[linear.weight] = (None, "model")
        if linear.bias is not None:
            specs[linear.bias] = (None,)

    for block in transformer.transformer_blocks:
        attn = block.attn
        for lin in (attn.to_q, attn.to_k, attn.to_v):
            col(lin)
        row(attn.to_out[0])
        for lin in (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj):
            col(lin)
        row(attn.to_add_out)
        col(block.ff.net[0].proj)
        row(block.ff.net[2])
        col(block.ff_context.net[0].proj)
        row(block.ff_context.net[2])

    for block in transformer.single_transformer_blocks:
        attn = block.attn
        for lin in (attn.to_q, attn.to_k, attn.to_v):
            col(lin)
        col(block.proj_mlp)
        row(block.proj_out)

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

    # ---- multi-chip sharding (Megatron tensor-parallel) --------------------
    # Mesh shapes by total device count: (batch, model). n300 -> (1, 2). The
    # repo's get_mesh_shape_for_device_count helper only covers >=8 chips, so
    # n300's 2-chip shape is provided here (mirrors the Mochi loader pattern).
    MESH_NAMES = ("batch", "model")
    MESH_SHAPES = {
        2: (1, 2),  # n300
        8: (1, 8),  # n300-llmbox / WH QuietBox
        32: (4, 8),  # WH galaxy
    }

    def get_mesh_config(self, num_devices: int):
        """Return ((batch, model) mesh shape, mesh names) for the active component."""
        if num_devices not in self.MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(self.MESH_SHAPES)}."
            )
        return self.MESH_SHAPES[num_devices], self.MESH_NAMES

    def load_shard_spec(self, model):
        """Return {tensor: partition_spec} for the active component.

        The transformer is sharded Megatron tensor-parallel on the "model" axis;
        the text encoder and VAE are returned unsharded here (the VAE's conv
        bottleneck needs a different scheme; the TE is a separate sub-problem).
        """
        if self._variant == ModelVariant.TRANSFORMER:
            return _shard_transformer_specs(model.transformer)
        return None
