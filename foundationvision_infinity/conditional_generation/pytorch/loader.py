# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FoundationVision Infinity (bitwise autoregressive text-to-image) loader.

Infinity (arXiv:2412.04431) is a *bitwise visual autoregressive* text-to-image
model. It is not a single forward pass but a pipeline:

    text encoder (Flan-T5-XL)  ->  Infinity transformer (the AR model)  ->  bitwise VAE decoder

The Infinity transformer is the compute-dominant component (the analog of a
diffusion model's denoiser) and is what this loader brings up on device. The
text encoder and VAE decoder stay in host Python during real generation; the
transformer is the part that must run on the accelerator.

This loader vendors the upstream model code (github.com/FoundationVision/Infinity,
MIT licensed) under ``src/infinity`` and provides a pure-PyTorch ``flash_attn``
shim under ``src/flash_attn`` so the model imports and runs without the CUDA-only
``flash_attn`` package. The model is built exactly as upstream's
``tools/run_infinity.py:load_infinity`` builds it (``customized_flash_attn=False``
so self-attention takes the ``scaled_dot_product_attention`` path), and the
public ``infinity_125M_256x256.pth`` checkpoint is loaded (0 missing / 0
unexpected keys).

The transformer's training-style ``forward`` is exposed through a thin
``_InfinityForward`` wrapper whose ``forward`` takes only tensors
(``x_BLC_wo_prefix`` teacher-forcing embeddings and the ``kv_compact`` text
features); the multi-scale schedule and constant text length are baked in so the
graph is fully traceable. A single forward over the 256x256 (pn=0.06M) schedule
produces bitwise logits of shape ``[B, 521, 32]``.
"""

import os
import sys

import torch
import torch.nn as nn

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

# Make the vendored ``infinity`` package and the ``flash_attn`` shim importable.
# ``src`` is prepended so the local shim wins over any (CUDA-only) flash_attn.
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


class ModelVariant(StrEnum):
    """Available Infinity variants."""

    INFINITY_125M_256 = "125M_256x256"


from dataclasses import dataclass


@dataclass
class InfinityConfig(ModelConfig):
    """Configuration for an Infinity transformer variant."""

    checkpoint_file: str = "infinity_125M_256x256.pth"
    # VAE bitwise codebook dimension (== transformer input feature dim d_vae).
    codebook_dim: int = 16
    # Transformer architecture (matches upstream model_type registry).
    depth: int = 12
    embed_dim: int = 768
    num_heads: int = 8
    block_chunks: int = 4
    # Resolution / schedule selectors.
    pn: str = "0.06M"
    h_div_w: float = 1.0
    # Text encoder (Flan-T5-XL) hidden size and the fixed number of text tokens
    # used to size the cross-attention key/value sequence for bringup.
    text_channels: int = 2048
    text_len: int = 128


_VARIANTS = {
    ModelVariant.INFINITY_125M_256: InfinityConfig(
        pretrained_model_name="FoundationVision/Infinity",
        checkpoint_file="infinity_125M_256x256.pth",
        codebook_dim=16,
        depth=12,
        embed_dim=768,
        num_heads=8,
        block_chunks=4,
        pn="0.06M",
        h_div_w=1.0,
        text_channels=2048,
        text_len=128,
    ),
}

DEFAULT_VARIANT = ModelVariant.INFINITY_125M_256


class _StubLFQ:
    """Stand-in for the VAE quantizer's lookup-free-quantization module.

    The Infinity transformer only reads ``vae.embed_dim``, ``vae.vocab_size`` and
    ``vae.quantizer.lfq.mask`` at construction time (the VAE itself is not a
    submodule and contributes no weights). The bitwise self-correction ``mask`` is
    the standard power-of-two bit mask; it is unused by the forward pass exercised
    here but is provided for fidelity.
    """

    def __init__(self, codebook_dim: int):
        self.mask = 2 ** torch.arange(codebook_dim)


class _StubQuantizer:
    def __init__(self, codebook_dim: int):
        self.lfq = _StubLFQ(codebook_dim)


class _StubVAE:
    """Lightweight VAE stand-in exposing only the attributes Infinity reads."""

    def __init__(self, codebook_dim: int):
        self.embed_dim = codebook_dim
        self.vocab_size = 2 ** codebook_dim
        self.quantizer = _StubQuantizer(codebook_dim)


class _InfinityForward(nn.Module):
    """Tensor-only wrapper around the Infinity transformer's forward.

    The underlying ``Infinity.forward(label_B_or_BLT, x_BLC_wo_prefix,
    scale_schedule)`` takes a text-condition tuple and a (constant) multi-scale
    schedule. This wrapper exposes a ``forward(x_BLC_wo_prefix, kv_compact)`` that
    takes only tensors, rebuilding the condition tuple and slotting in the baked
    schedule / text length, so the resulting graph is traceable on device.
    """

    def __init__(self, model: nn.Module, scale_schedule, text_len: int):
        super().__init__()
        self.model = model
        # List of (1, h, w) tuples — a python constant, baked into the graph.
        self.scale_schedule = scale_schedule
        self.text_len = int(text_len)

    def forward(self, x_BLC_wo_prefix, kv_compact):
        n = self.text_len
        cu_seqlens_k = torch.tensor(
            [0, n], dtype=torch.int32, device=kv_compact.device
        )
        # label_B_or_BLT = (kv_compact, lens, cu_seqlens_k, max_seqlen_k)
        label = (kv_compact, [n], cu_seqlens_k, n)
        return self.model(label, x_BLC_wo_prefix, self.scale_schedule)


class ModelLoader(ForgeModel):
    """Loader for the FoundationVision Infinity transformer."""

    _VARIANTS = _VARIANTS
    DEFAULT_VARIANT = DEFAULT_VARIANT

    def __init__(self, variant=None):
        super().__init__(variant)
        self._scale_schedule = None
        self._seq_len = None

    @classmethod
    def _get_model_info(cls, variant=None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="foundationvision_infinity",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_scale_schedule(self, cfg):
        """Resolve the multi-scale (1, h, w) schedule for this variant.

        Upstream inference forces the temporal dim to 1 (run_infinity.py), and
        the precomputed RoPE-2D grid is keyed by these (1, h, w) tuples, so the
        schedule passed to the model must use t=1.
        """
        from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

        scales = dynamic_resolution_h_w[cfg.h_div_w][cfg.pn]["scales"]
        scale_schedule = [(1, h, w) for (_, h, w) in scales]
        seq_len = int(sum(t * h * w for (t, h, w) in scale_schedule))
        self._scale_schedule = scale_schedule
        self._seq_len = seq_len
        return scale_schedule, seq_len

    def load_model(self, **kwargs):
        """Build the Infinity transformer, load weights, return a forward wrapper.

        The model is run fully in float32: Infinity uses mixed precision on CUDA
        (only the transformer blocks are cast to bf16, with autocast bridging the
        fp32 embeddings/head). Replicating that selective cast + autocast on the
        compile path is fragile and unnecessary for a 125M model, so weights are
        kept in float32 (deterministic, ~0.5 GB, fits a single n150 chip).
        """
        from huggingface_hub import hf_hub_download
        from infinity.models.infinity import Infinity

        cfg = self._variant_config
        scale_schedule, _ = self._build_scale_schedule(cfg)
        vae = _StubVAE(cfg.codebook_dim)

        model = Infinity(
            vae_local=vae,
            text_channels=cfg.text_channels,
            text_maxlen=512,
            shared_aln=True,
            raw_scale_schedule=None,
            checkpointing="full-block",
            customized_flash_attn=False,
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=False,
            add_lvl_embeding_only_first_block=1,
            use_bit_label=1,
            rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2,
            pn=cfg.pn,
            apply_spatial_patchify=0,
            inference_mode=True,
            train_h_div_w_list=[cfg.h_div_w],
            depth=cfg.depth,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            drop_path_rate=0.1,
            mlp_ratio=4,
            block_chunks=cfg.block_chunks,
        )

        ckpt_path = hf_hub_download(cfg.pretrained_model_name, cfg.checkpoint_file)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

        model.eval()
        model.requires_grad_(False)
        # Disable classifier-free-guidance condition dropping so the forward is
        # deterministic (otherwise random.random() gating would diverge between
        # the CPU reference and device runs).
        model.cond_drop_rate = 0.0

        return _InfinityForward(model, scale_schedule, cfg.text_len)

    def load_inputs(self, **kwargs):
        """Build sample inputs for the transformer forward.

        Returns a dict with:
          * ``x_BLC_wo_prefix`` — teacher-forcing bitwise-embedding sequence,
            shape [B, seq_len - 1, codebook_dim]. The model prepends a SOS token,
            so the full sequence length matches sum(h*w) over the schedule.
          * ``kv_compact`` — Flan-T5-XL text features, shape [text_len, text_channels].

        Random (seeded) tensors are used: this validates that the transformer
        compiles and matches the CPU reference (the same inputs feed both). Real
        text features would come from running Flan-T5-XL in host Python; that
        encoder is not part of the on-device component.
        """
        cfg = self._variant_config
        if self._seq_len is None:
            self._build_scale_schedule(cfg)
        seq_len = self._seq_len

        gen = torch.Generator().manual_seed(0)
        x_BLC_wo_prefix = torch.randn(
            1, seq_len - 1, cfg.codebook_dim, generator=gen, dtype=torch.float32
        )
        kv_compact = torch.randn(
            cfg.text_len, cfg.text_channels, generator=gen, dtype=torch.float32
        )
        return {
            "x_BLC_wo_prefix": x_BLC_wo_prefix,
            "kv_compact": kv_compact,
        }
