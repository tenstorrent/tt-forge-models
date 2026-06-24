# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Infinity bitwise autoregressive transformer loader implementation.

`FoundationVision/Infinity` (arXiv:2412.04431) is a bitwise *visual*
autoregressive text-to-image model. It is NOT a `diffusers`/`transformers`
pipeline; the modeling code lives in the upstream GitHub repo
(`github.com/FoundationVision/Infinity`) and is vendored under
`infinity/_vendor/infinity_ar/`. The only patch applied to the upstream code is
a dense torch-SDPA fallback for the CUDA-only `flash_attn` kernels (see
`_vendor/infinity_ar/models/basic.py`); the inference path already runs with
`customized_flash_attn=False`, so self-attention is unaffected and only the text
cross-attention uses the dense varlen fallback.

The pipeline has three components, each with its own loader directory:
text encoder (Flan-T5-XL, CPU-side preprocessing), the **Infinity transformer**
(this loader — the compute-dominant autoregressive denoiser analog and the
Tenstorrent target), and the bitwise VAE (`../../vae`).

This loader exposes the transformer's single, teacher-forced forward pass
(`Infinity.forward(label, x_BLC_wo_prefix, scale_schedule) -> logits_BLV`),
which runs all transformer blocks in one graph and is the natural unit for an
on-device compile/PCC test. The autoregressive sampling loop, VAE decode, and
text encoding stay in host Python in the composite pipeline.

The default variant is the smallest released checkpoint (125M, `infinity_layer12`,
256x256 / 0.06M-pixel schedule), paired with the d16 bitwise VAE
(`codebook_dim=16`, so the model word-embedding/head dims are 16 / 2*16).
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

# Make the vendored upstream package importable as `infinity_ar`.
_VENDOR = Path(__file__).resolve().parents[2] / "_vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))


class ModelVariant(StrEnum):
    """Available Infinity transformer variants."""

    INFINITY_125M = "125M"


# Per-variant architecture + checkpoint wiring. `vae_codebook_dim` selects the
# matching bitwise VAE (the 125M checkpoint was trained against the d16 VAE,
# fixing word_embed in-dim = 16 and head out-dim = 2*16).
_ARCH = {
    ModelVariant.INFINITY_125M: dict(
        model_kwargs=dict(
            depth=12, embed_dim=768, num_heads=8,
            drop_path_rate=0.1, mlp_ratio=4, block_chunks=4,
        ),
        weights_file="infinity_125M_256x256.pth",
        vae_file="infinity_vae_d16.pth",
        vae_codebook_dim=16,
        pn="0.06M",
        h_div_w=1.0,
    ),
}

_HF_REPO = "FoundationVision/Infinity"


class _InfinitySingleForward(torch.nn.Module):
    """Wrap `Infinity` so its forward consumes only tensors.

    The upstream `Infinity.forward` takes a mixed-type conditioning tuple
    `(kv_compact, lens, cu_seqlens_k, max_seqlen_k)` and a `scale_schedule`
    list. The bringup runs batch size 1 with a fixed text length, so the
    non-tensor pieces (`lens`, `max_seqlen_k`, `cu_seqlens_k`, `scale_schedule`)
    are constants captured here, and the device-facing forward takes the two
    real tensors only.
    """

    def __init__(self, model, scale_schedule, text_len):
        super().__init__()
        self.model = model
        self.scale_schedule = scale_schedule
        self.text_len = int(text_len)
        self.register_buffer(
            "cu_seqlens_k",
            torch.tensor([0, self.text_len], dtype=torch.int32),
            persistent=False,
        )

    def forward(self, kv_compact, x_BLC_wo_prefix):
        label = (kv_compact, [self.text_len], self.cu_seqlens_k, self.text_len)
        return self.model(label, x_BLC_wo_prefix, self.scale_schedule)


class ModelLoader(ForgeModel):
    """Loader for the Infinity bitwise autoregressive transformer."""

    _VARIANTS = {
        ModelVariant.INFINITY_125M: ModelConfig(
            pretrained_model_name="FoundationVision/Infinity",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INFINITY_125M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._scale_schedule = None
        self._text_len = 32  # number of conditioning text tokens (Flan-T5 output)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="infinity",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _scale_schedule_for(self, arch):
        """Resolve the (1, h, w) per-scale schedule for the variant resolution."""
        import numpy as np
        from infinity_ar.utils.dynamic_resolution import (
            dynamic_resolution_h_w,
            h_div_w_templates,
        )

        tmpl = h_div_w_templates[
            np.argmin(np.abs(h_div_w_templates - arch["h_div_w"]))
        ]
        scales = dynamic_resolution_h_w[tmpl][arch["pn"]]["scales"]
        # Cast to plain Python ints (the source values are numpy ints) so the
        # schedule used for slice/pointer arithmetic stays torch.compile-traceable.
        return [(1, int(h), int(w)) for (_, h, w) in scales]

    def _build_vae(self, arch, device="cpu"):
        """Build the matching bitwise VAE. Only its structural config (codebook
        dim) feeds the transformer; the transformer's forward never calls the
        VAE, so VAE weights do not affect this test, but loading the released
        checkpoint guarantees the config matches the trained model exactly."""
        from infinity_ar.models.bsq_vae.vae import vae_model

        vae_path = hf_hub_download(repo_id=_HF_REPO, filename=arch["vae_file"])
        cd = arch["vae_codebook_dim"]
        vae = vae_model(
            vae_path, "dynamic", cd, 2 ** cd, patch_size=16,
            encoder_ch_mult=[1, 2, 4, 4, 4], decoder_ch_mult=[1, 2, 4, 4, 4],
            test_mode=True,
        ).to(device)
        return vae

    def load_model(self, dtype_override: Optional[torch.dtype] = None):
        """Build the Infinity transformer, load released weights, and wrap it
        for a tensor-only single forward pass."""
        from infinity_ar.models.infinity import Infinity

        arch = _ARCH[self._variant]
        self._scale_schedule = self._scale_schedule_for(arch)
        vae = self._build_vae(arch)

        model = Infinity(
            vae_local=vae,
            text_channels=2048,
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
            pn=arch["pn"],
            apply_spatial_patchify=0,
            inference_mode=True,
            train_h_div_w_list=[1.0],
            **arch["model_kwargs"],
        )
        # Disable classifier-free-guidance conditioning dropout: it is a training
        # augmentation that randomly (and in-place) zeroes conditioning, which
        # would make the forward nondeterministic. Inference uses no dropout.
        model.cond_drop_rate = 0.0
        model.eval()

        weights_path = hf_hub_download(
            repo_id=_HF_REPO, filename=arch["weights_file"]
        )
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

        wrapped = _InfinitySingleForward(
            model, self._scale_schedule, self._text_len
        )
        wrapped.eval()
        # Run uniformly in bfloat16 (the TT-native data format). Infinity ships
        # hardcoded fp32 islands for CUDA-autocast execution; the vendored code
        # has those `.float()` activation upcasts removed in the executed path so
        # the whole graph stays a single dtype and avoids mixed-dtype norm/matmul
        # errors on device.
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        wrapped = wrapped.to(dtype=dtype)
        return wrapped

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Synthetic, seeded inputs for one teacher-forced forward.

        - `kv_compact`: Flan-T5 conditioning features, shape (text_len, 2048).
        - `x_BLC_wo_prefix`: the per-token bitwise VAE embeddings minus the SOS
          prefix, shape (B=1, L-1, codebook_dim), where L = sum(h*w) over scales.
        """
        if self._scale_schedule is None:
            self._scale_schedule = self._scale_schedule_for(_ARCH[self._variant])
        arch = _ARCH[self._variant]
        codebook_dim = arch["vae_codebook_dim"]
        total_len = sum(t * h * w for (t, h, w) in self._scale_schedule)

        gen = torch.Generator().manual_seed(0)
        kv_compact = torch.randn(self._text_len, 2048, generator=gen)
        x_BLC_wo_prefix = torch.randn(1, total_len - 1, codebook_dim, generator=gen)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        kv_compact = kv_compact.to(dtype=dtype)
        x_BLC_wo_prefix = x_BLC_wo_prefix.to(dtype=dtype)

        return {
            "kv_compact": kv_compact,
            "x_BLC_wo_prefix": x_BLC_wo_prefix,
        }
