# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 DiT transformer loader for text-to-video (+audio) generation.

Unlike the diffusers-based ``ltx2`` family loader, this loader drives the
*native* ``ltx_core`` transformer code vendored under ``src/ltx_core/`` (see
``src/SRC_VENDORED_FROM.txt``). The 22B LTX-2.3 audio-video DiT is built
straight from the checkpoint's embedded transformer config via
``LTXModelConfigurator.from_config`` with random weights -- no checkpoint is
downloaded and no HF pipeline is instantiated.

Repository: https://github.com/Lightricks/LTX-2
Weights:    https://huggingface.co/Lightricks/LTX-2.3

The native ``LTXModel.forward`` takes structured ``Modality`` objects rather
than plain tensors:

    forward(video: Modality | None, audio: Modality | None,
            perturbations: BatchedPerturbationConfig) -> (video_out, audio_out)

so ``load_model`` returns an ``nn.Module`` wrapper whose ``forward(*tensors)``
rebuilds the ``Modality`` objects and the (no-op) perturbation config, calls
the underlying model, and returns the video output tensor. ``load_inputs``
returns the matching plain tensors in the wrapper's forward-arg order.

Both variants build the SAME architecture from the SAME embedded config; they
differ only in the checkpoint they would load (which this scaffold does NOT do):

    Fast -> ltx-2.3-22b-distilled-1.1.safetensors
    Pro  -> ltx-2.3-22b-dev.safetensors

NOTE: the full 48-layer model is ~21B params -- host-CPU instantiation is
infeasible. Treat the transformer as derived / not-CPU-instantiated, exactly
like the diffusers ``ltx2`` reference. The reduced-layer CPU forward used to
validate the plumbing overrides ``num_layers`` to a small value.
"""

import os
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

# Vendored ltx_core lives under src/; add it to sys.path before importing.
_SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ltx_core.guidance.perturbations import BatchedPerturbationConfig  # noqa: E402
from ltx_core.model.transformer.modality import Modality  # noqa: E402
from ltx_core.model.transformer.model_configurator import (  # noqa: E402
    LTXModelConfigurator,
)

_HF_REPO = "Lightricks/LTX-2.3"

# ── Embedded transformer config ─────────────────────────────────────────────
# Extracted from the LTX-2.3 22B checkpoint's safetensors header (the
# "transformer" sub-dict of the model config). ``LTXModelConfigurator.from_config``
# reads ONLY ``config["transformer"]`` (both directly and via
# ``_build_caption_projections``), so the full dict here just nests that sub-dict
# under the "transformer" key. ``caption_proj_before_connector=True`` puts the
# caption projection in the text encoder (22B path), so no projection module is
# built inside the transformer -- the cross-attention context arrives already at
# ``cross_attention_dim``.
_TRANSFORMER_CONFIG = {
    "_class_name": "AVTransformer3DModel",
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 128,
    "attention_type": "default",
    "caption_channels": 3840,
    "cross_attention_dim": 4096,
    "double_self_attention": False,
    "dropout": 0.0,
    "in_channels": 128,
    "norm_elementwise_affine": False,
    "norm_eps": 1e-06,
    "norm_num_groups": 32,
    "num_attention_heads": 32,
    "num_embeds_ada_norm": 1000,
    "num_layers": 48,
    "num_vector_embeds": None,
    "only_cross_attention": False,
    "cross_attention_norm": True,
    "out_channels": 128,
    "upcast_attention": False,
    "use_linear_projection": False,
    "qk_norm": "rms_norm",
    "standardization_norm": "rms_norm",
    "positional_embedding_type": "rope",
    "positional_embedding_theta": 10000.0,
    "positional_embedding_max_pos": [20, 2048, 2048],
    "timestep_scale_multiplier": 1000,
    "av_ca_timestep_scale_multiplier": 1000.0,
    "causal_temporal_positioning": True,
    "audio_num_attention_heads": 32,
    "audio_attention_head_dim": 64,
    "use_audio_video_cross_attention": True,
    "share_ff": False,
    "audio_out_channels": 128,
    "audio_cross_attention_dim": 2048,
    "audio_positional_embedding_max_pos": [20],
    "av_cross_ada_norm": True,
    "use_embeddings_connector": True,
    "connector_attention_head_dim": 128,
    "connector_num_attention_heads": 32,
    "connector_num_layers": 8,
    "connector_positional_embedding_max_pos": [4096],
    "connector_num_learnable_registers": 128,
    "connector_norm_output": True,
    "use_middle_indices_grid": True,
    "apply_gated_attention": True,
    "connector_apply_gated_attention": True,
    "caption_projection_first_linear": False,
    "caption_projection_second_linear": False,
    "caption_proj_input_norm": False,
    "connector_learnable_registers_std": 1,
    "caption_proj_before_connector": True,
    "audio_connector_attention_head_dim": 64,
    "audio_connector_num_attention_heads": 32,
    "cross_attention_adaln": True,
    "text_encoder_norm_type": "per_token_rms",
    "rope_type": "split",
    "frequencies_precision": "float64",
}
_MODEL_CONFIG = {"transformer": _TRANSFORMER_CONFIG}

# ── Derived feature dims (read off _TRANSFORMER_CONFIG) ──────────────────────
# model.py: inner_dim = num_attention_heads * attention_head_dim. The cross-attn
# context (attn2.context_dim) == cross_attention_dim. transformer_args.prepare
# reshapes context to (B, -1, inner_dim); inner_dim == cross_attention_dim here,
# so 4096 video / 2048 audio is consistent. (modality.py / transformer.py /
# transformer_args.py.)
_IN_CHANNELS = _TRANSFORMER_CONFIG["in_channels"]  # latent feature dim D = 128
_AUDIO_IN_CHANNELS = 128  # audio_in_channels default (model.py)
_VIDEO_CTX_DIM = (
    _TRANSFORMER_CONFIG["num_attention_heads"]
    * _TRANSFORMER_CONFIG["attention_head_dim"]
)  # 4096 == cross_attention_dim
_AUDIO_CTX_DIM = (
    _TRANSFORMER_CONFIG["audio_num_attention_heads"]
    * _TRANSFORMER_CONFIG["audio_attention_head_dim"]
)  # 2048 == audio_cross_attention_dim

# Minimal valid sequence dims for a reduced-layer CPU sanity forward.
_VIDEO_TOKENS = 4
_AUDIO_TOKENS = 4
_CTX_SEQ = 8

# variant -> intended checkpoint filename (NOT loaded by this scaffold).
_VARIANT_CHECKPOINT = {
    "Fast": "ltx-2.3-22b-distilled-1.1.safetensors",
    "Pro": "ltx-2.3-22b-dev.safetensors",
}


class ModelVariant(StrEnum):
    LTX2_3_FAST = "Fast"
    LTX2_3_PRO = "Pro"


# ── Tensors-only wrapper ─────────────────────────────────────────────────────
class _LTXModelWrapper(torch.nn.Module):
    """Wrap the native ``LTXModel`` (which takes ``Modality`` objects) in a
    plain-tensor ``forward`` so the bringup harness can trace it.

    The non-tensor structural argument (the no-op perturbation config) is built
    inside ``forward`` from the batch size; only tensors cross the boundary.
    Returns the video output tensor (the audio output is computed but dropped).
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        video_latent,
        video_sigma,
        video_timesteps,
        video_positions,
        video_context,
        audio_latent,
        audio_sigma,
        audio_timesteps,
        audio_positions,
        audio_context,
    ):
        video = Modality(
            latent=video_latent,
            sigma=video_sigma,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_context,
        )
        audio = Modality(
            latent=audio_latent,
            sigma=audio_sigma,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_context,
        )
        perturbations = BatchedPerturbationConfig.empty(video_latent.shape[0])
        video_out, _audio_out = self.model(video, audio, perturbations)
        return video_out


class ModelLoader(ForgeModel):
    """LTX-2.3 22B audio-video DiT transformer loader (Fast / Pro variants)."""

    _VARIANTS = {v: ModelConfig(pretrained_model_name=_HF_REPO) for v in ModelVariant}
    DEFAULT_VARIANT = ModelVariant.LTX2_3_FAST

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LTX2_3",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, num_layers=None, **kwargs):
        """Build the native LTXModel from the embedded config with RANDOM
        weights and wrap it for a tensors-only forward.

        Both variants build the same architecture; ``_VARIANT_CHECKPOINT`` records
        the checkpoint each would load (not loaded here). ``num_layers`` is an
        override for CPU sanity checks only -- the full 48-layer model is ~21B
        params and cannot be instantiated on host RAM.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        config = _MODEL_CONFIG
        if num_layers is not None:
            config = {"transformer": {**_TRANSFORMER_CONFIG, "num_layers": num_layers}}

        base = LTXModelConfigurator.from_config(config)
        base = base.to(dtype).eval()

        self.model = _LTXModelWrapper(base)
        if dtype_override is not None:
            self.model = self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Synthetic plain tensors at minimal valid shapes, returned in the
        wrapper's forward-arg order (video block then audio block).

        Shapes follow ``Modality`` (modality.py) + the args preprocessors
        (transformer_args.py): latent (B, T, D=in_channels); context
        (B, ctx_seq, inner_dim) where inner_dim == heads*head_dim == the cross-
        attention dim (4096 video / 2048 audio, caption projection lives in the
        text encoder for 22B); positions (B, n_pos_dims, T, 2) with n_pos_dims=3
        video / 1 audio and last dim = [start, end) patch bounds because
        use_middle_indices_grid=True; timesteps (B, T); sigma (B,).
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        def _positions(n_pos_dims, tokens):
            # [start, end) integer patch bounds: end = start + 1.
            start = torch.arange(tokens, dtype=dtype).view(1, 1, tokens, 1)
            start = start.expand(batch_size, n_pos_dims, tokens, 1)
            return torch.cat([start, start + 1], dim=-1)

        return [
            # video
            torch.randn(batch_size, _VIDEO_TOKENS, _IN_CHANNELS, dtype=dtype),
            torch.full((batch_size,), 0.5, dtype=dtype),
            torch.full((batch_size, _VIDEO_TOKENS), 0.5, dtype=dtype),
            _positions(3, _VIDEO_TOKENS),
            torch.randn(batch_size, _CTX_SEQ, _VIDEO_CTX_DIM, dtype=dtype),
            # audio
            torch.randn(batch_size, _AUDIO_TOKENS, _AUDIO_IN_CHANNELS, dtype=dtype),
            torch.full((batch_size,), 0.5, dtype=dtype),
            torch.full((batch_size, _AUDIO_TOKENS), 0.5, dtype=dtype),
            _positions(1, _AUDIO_TOKENS),
            torch.randn(batch_size, _CTX_SEQ, _AUDIO_CTX_DIM, dtype=dtype),
        ]

    def unpack_forward_output(self, output):
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    # ── Multichip tensor-parallel plan (Megatron 1D on the model axis) ──────
    def get_mesh_config(self, num_devices: int):
        """Return ((1, num_devices), ("batch", "model")) for Megatron-style TP."""
        return (1, num_devices), ("batch", "model")

    def load_shard_spec(self, model):
        """Megatron-style TP map over the transformer blocks. Non-sharded dim is
        ``None`` (replicated).

        Module names are taken from the vendored ``BasicAVTransformerBlock``
        (transformer.py): per-block attentions ``attn1`` / ``attn2`` (video),
        ``audio_attn1`` / ``audio_attn2`` (audio), and the AV cross-attentions
        ``audio_to_video_attn`` / ``video_to_audio_attn``; feed-forwards ``ff``
        / ``audio_ff``. Each ``Attention`` exposes ``to_q`` / ``to_k`` / ``to_v``
        and an output projection; ``FeedForward`` wraps an ``nn.Sequential``
        ``net``. The exact submodule names of ``Attention`` / ``FeedForward``
        were not fully inspected, so this is written DEFENSIVELY: any missing
        attribute is skipped. Column-parallel q/k/v + row-parallel out is the
        standard Megatron split.
        """
        shard_specs = {}
        wrapped = getattr(model, "model", model)
        blocks = getattr(wrapped, "transformer_blocks", None)
        if blocks is None:
            return shard_specs

        attn_names = (
            "attn1",
            "attn2",
            "audio_attn1",
            "audio_attn2",
            "audio_to_video_attn",
            "video_to_audio_attn",
        )
        ff_names = ("ff", "audio_ff")

        def _w(module, attr):
            sub = getattr(module, attr, None)
            return getattr(sub, "weight", None) if sub is not None else None

        for block in blocks:
            for attn_name in attn_names:
                attn = getattr(block, attn_name, None)
                if attn is None:
                    continue
                # Column-parallel q/k/v projections. ``to_gate_logits`` is a
                # per-head gate (out dim == heads, verified shape (heads, dim));
                # its output is applied per-head to the head-sharded attn output
                # (see ops.PytorchGatedAttention), so it MUST be sharded on the
                # head/output dim to match — a replicated gate shape-mismatches.
                for proj in ("to_q", "to_k", "to_v", "to_gate_logits"):
                    w = _w(attn, proj)
                    if w is not None:
                        shard_specs[w] = ("model", None)
                # Row-parallel output projection. ltx_core's Attention may expose
                # the output projection under one of these names; try each.
                for out_name in ("to_out", "out_proj", "proj_out"):
                    out = getattr(attn, out_name, None)
                    if out is None:
                        continue
                    # to_out is sometimes an nn.Sequential/ModuleList.
                    if hasattr(out, "weight"):
                        shard_specs[out.weight] = (None, "model")
                    elif hasattr(out, "__getitem__"):
                        try:
                            shard_specs[out[0].weight] = (None, "model")
                        except (IndexError, AttributeError, TypeError):
                            pass
                    break
            for ff_name in ff_names:
                ff = getattr(block, ff_name, None)
                if ff is None or not hasattr(ff, "net"):
                    continue
                net = ff.net
                # net[0] (or its .proj) is the up-projection (column); net[-1] is
                # the down-projection (row).
                first = net[0]
                first_w = getattr(getattr(first, "proj", first), "weight", None)
                if first_w is not None:
                    shard_specs[first_w] = ("model", None)
                last_w = getattr(net[-1], "weight", None)
                if last_w is not None:
                    shard_specs[last_w] = (None, "model")
        return shard_specs
