# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image pipeline component loader for text-to-image generation.

Qwen-Image (Qwen) loads as a ``QwenImagePipeline`` whose ``model_index.json``
enumerates three compilable sub-models. Each is exposed here as its own
``ModelVariant`` with a tensors-only wrapper, so every component can be brought
up independently on a single device (the small VAE) or a tensor-parallel mesh
(the weight-bound transformer / text encoder).

Repository: https://github.com/QwenLM/Qwen-Image
Weights:    https://huggingface.co/Qwen/Qwen-Image

Components (params measured by config unless noted "est"):

    transformer   QwenImageTransformer2DModel        ~20B (est)  diffusers
    text_encoder  Qwen2_5_VLForConditionalGeneration ~7.6B (est) transformers
    vae           AutoencoderKLQwenImage             ~0.25B      diffusers

The 40.9 GiB (bf16) MMDiT transformer exceeds every single TT chip, so the
transformer and the 7.6B Qwen2.5-VL text encoder are scaffolded
``tensor_parallel`` (Megatron 1D on the model axis); only the VAE fits a single
device. I/O shapes were captured from one CPU pipeline pass at 256x256
(``num_inference_steps=2``); the VAE latent shape is derived analytically.
"""

from typing import Optional

import torch
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from transformers import Qwen2_5_VLForConditionalGeneration

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

_HF_REPO = "Qwen/Qwen-Image"

# ── Shape constants (captured from a 256x256 CPU pipeline pass) ─────────────
# Transformer (transformer/config.json): 60 blocks, 24 heads x 128, inner 3072.
_IN_CHANNELS = 64  # patchified latent channels (16 z_dim x 2x2 patch)
_JOINT_ATTENTION_DIM = 3584  # == Qwen2.5-VL hidden_size (text stream width)
# 256x256 -> latent 32x32 -> patch 2 -> 16x16 = 256 image tokens.
_IMG_FRAMES, _IMG_H, _IMG_W = 1, 16, 16
_IMG_SEQ = _IMG_FRAMES * _IMG_H * _IMG_W  # 256
# img_shapes is a pinned structural arg (per-sample list of (F, H, W) for RoPE).
_IMG_SHAPES = [[(_IMG_FRAMES, _IMG_H, _IMG_W)]]
_TXT_SEQ = 13  # text tokens fed to the transformer (post pipeline trim)

# Text encoder (Qwen2.5-VL text_config): hidden 3584, 28 layers, GQA 28h/4kv.
_TE_HIDDEN = 3584
_TE_VOCAB_SIZE = 152064
_TE_SEQ_LEN = 47  # captured prompt length (template + caption)

# VAE (AutoencoderKLQwenImage): 3D causal-conv autoencoder, z_dim 16.
# 256x256 image -> latent (z_dim, T=1, 32, 32); decode -> pixel space.
_VAE_Z_DIM = 16
_VAE_LATENT_T = 1
_VAE_LATENT_HW = 32

# Captured per-component I/O spec (forward arg order == load_inputs() order).
_COMPONENT_IO_SPEC = {
    "transformer": {  # derived params ~20B (not CPU-instantiated for sizing)
        "inputs": [
            ("hidden_states", "float", (1, _IMG_SEQ, _IN_CHANNELS)),
            ("timestep", "float", (1,)),
            ("encoder_hidden_states", "float", (1, _TXT_SEQ, _JOINT_ATTENTION_DIM)),
        ],
        "pinned": {
            "img_shapes": _IMG_SHAPES,
            "encoder_hidden_states_mask": None,
            "txt_seq_lens": None,
            "guidance": None,
            "return_dict": False,
        },
        "output": "out[0] -> (1, 256, 64)",
    },
    "text_encoder": {  # Qwen2.5-VL ~7.6B
        "inputs": [
            ("input_ids", "int", (1, _TE_SEQ_LEN)),
            ("attention_mask", "int", (1, _TE_SEQ_LEN)),
        ],
        "output": "hidden_states[-1] -> (1, 47, 3584)",
    },
    "vae": {  # validated 0.25B; latent derived analytically
        "inputs": [
            (
                "latent",
                "float",
                (1, _VAE_Z_DIM, _VAE_LATENT_T, _VAE_LATENT_HW, _VAE_LATENT_HW),
            )
        ],
        "output": "vae.decode(...)[0]",
    },
}


class ModelVariant(StrEnum):
    QWEN_IMAGE_TRANSFORMER = "Transformer"
    QWEN_IMAGE_TEXT_ENCODER = "TextEncoder"
    QWEN_IMAGE_VAE = "Vae"


# variant -> (component class, model_index subfolder)
_COMPONENT = {
    ModelVariant.QWEN_IMAGE_TRANSFORMER: (QwenImageTransformer2DModel, "transformer"),
    ModelVariant.QWEN_IMAGE_TEXT_ENCODER: (
        Qwen2_5_VLForConditionalGeneration,
        "text_encoder",
    ),
    ModelVariant.QWEN_IMAGE_VAE: (AutoencoderKLQwenImage, "vae"),
}


# ── Tensors-only wrappers (pin non-tensor structural args, unwrap outputs) ──
class _TransformerWrapper(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        out = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=None,
            timestep=timestep,
            img_shapes=_IMG_SHAPES,
            txt_seq_lens=None,
            guidance=None,
            return_dict=False,
        )
        return out[0]


class _TextEncoderWrapper(torch.nn.Module):
    """Qwen2.5-VL text path -> last hidden state (batch, seq, hidden)."""

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids, attention_mask):
        out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return out.hidden_states[-1]


class _VaeDecoderWrapper(torch.nn.Module):
    """Decode a latent to pixel space (deterministic decode, not forward)."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        return self.vae.decode(latent, return_dict=False)[0]


class ModelLoader(ForgeModel):
    """Qwen-Image pipeline component loader (one variant per compilable component)."""

    _VARIANTS = {v: ModelConfig(pretrained_model_name=_HF_REPO) for v in ModelVariant}
    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_TRANSFORMER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QwenImage",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load one Qwen-Image component (direct from_pretrained, wrapped to a
        tensors-only forward)."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        cls, subfolder = _COMPONENT[self._variant]
        base = cls.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder=subfolder,
            torch_dtype=dtype,
        ).eval()

        if self._variant == ModelVariant.QWEN_IMAGE_TRANSFORMER:
            self.model = _TransformerWrapper(base)
        elif self._variant == ModelVariant.QWEN_IMAGE_TEXT_ENCODER:
            self.model = _TextEncoderWrapper(base)
        elif self._variant == ModelVariant.QWEN_IMAGE_VAE:
            self.model = _VaeDecoderWrapper(base)
        else:  # pragma: no cover
            raise ValueError(f"Unknown variant {self._variant}")

        if dtype_override is not None:
            self.model = self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Synthetic tensors at captured shapes, returned as a list in the
        wrapper's forward arg order."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        v = self._variant

        if v == ModelVariant.QWEN_IMAGE_TRANSFORMER:
            return [
                torch.randn(batch_size, _IMG_SEQ, _IN_CHANNELS, dtype=dtype),
                torch.full((batch_size,), 1000.0, dtype=dtype),
                torch.randn(batch_size, _TXT_SEQ, _JOINT_ATTENTION_DIM, dtype=dtype),
            ]
        if v == ModelVariant.QWEN_IMAGE_TEXT_ENCODER:
            return [
                torch.randint(
                    0, _TE_VOCAB_SIZE, (batch_size, _TE_SEQ_LEN), dtype=torch.long
                ),
                torch.ones(batch_size, _TE_SEQ_LEN, dtype=torch.long),
            ]
        if v == ModelVariant.QWEN_IMAGE_VAE:
            return [
                torch.randn(
                    batch_size,
                    _VAE_Z_DIM,
                    _VAE_LATENT_T,
                    _VAE_LATENT_HW,
                    _VAE_LATENT_HW,
                    dtype=dtype,
                )
            ]
        raise ValueError(f"Unknown variant {v}")  # pragma: no cover

    def unpack_forward_output(self, output):
        if isinstance(output, (tuple, list)):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output

    # ── Multichip tensor-parallel plan (Megatron 1D on the model axis) ──────
    # (batch, model) mesh by device count. The model (TP) axis is capped at 4:
    # the Qwen2.5-VL text encoder has 28 query heads and 4 KV heads, so the TP
    # degree must divide 28 and stay <= the KV-head count — TP>4 makes the
    # attention reshape (B, S, heads*head_dim) -> (B, S, heads, head_dim)
    # unshardable and tt-mlir fails with "Could not apply propagated tensor
    # shardings". TP=4 maps 7 query + 1 KV head per device; the transformer's
    # 24 heads / 4 = 6 also divide cleanly, so both components share the mesh.
    _MESH_SHAPES = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (2, 4), 32: (8, 4)}

    def get_mesh_config(self, num_devices: int):
        """Return ((batch, model), ("batch", "model")) for Megatron-style TP.

        The two sharded components need different TP degrees:
          - Transformer is full multi-head attention (24 heads) → shard across
            all devices (TP = num_devices). TP=8 is also required to fit the
            joint-attention ttnn.concat in L1; TP=4 overflows it.
          - Text encoder is Qwen2.5-VL GQA (28 query / 4 KV heads) → TP must
            divide 28 and stay <= the KV-head count, so it is capped at 4
            (e.g. (2, 4) on 8 chips); TP>4 makes the head reshape unshardable.
        VAE replicates (``load_shard_spec`` returns an empty map for it).
        """
        if self._variant == ModelVariant.QWEN_IMAGE_TRANSFORMER:
            return (1, num_devices), ("batch", "model")

        if num_devices not in self._MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(self._MESH_SHAPES)}."
            )
        return self._MESH_SHAPES[num_devices], ("batch", "model")

    @staticmethod
    def _shard_transformer(model):
        """Megatron-style TP map for QwenImageTransformer2DModel. The dual-stream
        joint attention shards both the image (to_q/k/v, to_out) and text
        (add_q/k/v_proj, to_add_out) projections, plus both FeedForwards."""
        shard_specs = {}
        transformer = getattr(model, "transformer", model)
        for block in transformer.transformer_blocks:
            attn = block.attn
            # image stream
            shard_specs[attn.to_q.weight] = ("model", None)
            shard_specs[attn.to_k.weight] = ("model", None)
            shard_specs[attn.to_v.weight] = ("model", None)
            shard_specs[attn.to_out[0].weight] = (None, "model")
            # text stream (added projections)
            shard_specs[attn.add_q_proj.weight] = ("model", None)
            shard_specs[attn.add_k_proj.weight] = ("model", None)
            shard_specs[attn.add_v_proj.weight] = ("model", None)
            shard_specs[attn.to_add_out.weight] = (None, "model")
            # FeedForwards: net[0].proj is the up-proj (column), net[-1] the
            # down-proj (row), for both image and text MLPs.
            for mlp in (block.img_mlp, block.txt_mlp):
                shard_specs[mlp.net[0].proj.weight] = ("model", None)
                shard_specs[mlp.net[-1].weight] = (None, "model")
        return shard_specs

    @staticmethod
    def _shard_text_encoder(model):
        """Megatron-style GQA-TP map for the Qwen2.5-VL text encoder. KV
        projections are replicated (GQA fallback) — only q_proj/o_proj and the
        MLP gate/up/down are sharded."""
        shard_specs = {}
        te = getattr(model, "text_encoder", model)
        layers = te.model.language_model.layers
        for layer in layers:
            attn = layer.self_attn
            shard_specs[attn.q_proj.weight] = ("model", None)
            shard_specs[attn.o_proj.weight] = (None, "model")
            mlp = layer.mlp
            shard_specs[mlp.gate_proj.weight] = ("model", None)
            shard_specs[mlp.up_proj.weight] = ("model", None)
            shard_specs[mlp.down_proj.weight] = (None, "model")
        return shard_specs

    def load_shard_spec(self, model):
        """Megatron-style TP map. Non-sharded dim is ``None`` (replicated).

        Dispatches to the per-component sharding function. Only the weight-bound
        transformer / text encoder are sharded; the VAE replicates → empty map.
        """
        shard_fns = {
            ModelVariant.QWEN_IMAGE_TRANSFORMER: self._shard_transformer,
            ModelVariant.QWEN_IMAGE_TEXT_ENCODER: self._shard_text_encoder,
        }
        shard_fn = shard_fns.get(self._variant)
        return shard_fn(model) if shard_fn is not None else {}
