# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Infinity bitwise VAE (decoder) loader implementation.

`FoundationVision/Infinity` uses a multi-scale bitwise (BSQ) visual tokenizer.
This loader targets the **VAE decoder** — the convolutional network that maps the
quantized multi-scale latent `z` back to an RGB image — which is the part of the
tokenizer that runs on device in the composite text-to-image pipeline (the AR
transformer in `../../transformer` produces bit logits, which are dequantized to
`z`, then decoded here).

The modeling code is vendored under `infinity/_vendor/infinity_ar/` (see the
transformer loader for the vendoring/patch notes). The default variant is the
d16 tokenizer (`codebook_dim=16`), which pairs with the 125M transformer.

For 256x256 (0.06M-pixel) generation the decoder consumes a latent of shape
`[B, embed_dim=16, 16, 16]` and produces `[B, 3, 256, 256]`.
"""

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

_VENDOR = Path(__file__).resolve().parents[2] / "_vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

_HF_REPO = "FoundationVision/Infinity"

_ARCH = {
    # variant -> (vae_file, codebook_dim, latent_hw)
    "d16": dict(vae_file="infinity_vae_d16.pth", codebook_dim=16, latent_hw=16),
}


class ModelVariant(StrEnum):
    """Available Infinity VAE variants."""

    D16 = "d16"


class _VaeDecode(torch.nn.Module):
    """Expose the VAE decoder's `decode(z) -> image` as a tensor-in/tensor-out
    forward for the bringup harness."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z)


class ModelLoader(ForgeModel):
    """Loader for the Infinity bitwise VAE decoder."""

    _VARIANTS = {
        ModelVariant.D16: ModelConfig(
            pretrained_model_name="FoundationVision/Infinity",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.D16

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="infinity_vae",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_vae(self, device="cpu"):
        from infinity_ar.models.bsq_vae.vae import vae_model

        arch = _ARCH[str(self._variant)]
        vae_path = hf_hub_download(repo_id=_HF_REPO, filename=arch["vae_file"])
        cd = arch["codebook_dim"]
        vae = vae_model(
            vae_path, "dynamic", cd, 2 ** cd, patch_size=16,
            encoder_ch_mult=[1, 2, 4, 4, 4], decoder_ch_mult=[1, 2, 4, 4, 4],
            test_mode=True,
        ).to(device)
        return vae

    def load_model(self, dtype_override: Optional[torch.dtype] = None):
        vae = self._build_vae()
        wrapped = _VaeDecode(vae)
        wrapped.eval()
        # The decoder is a uniform-dtype conv stack (no fp32 islands), so it is
        # kept in float32: the TT compiler lowers the whole graph to its native
        # format consistently, and the CPU golden run stays fast (bf16 convs are
        # not optimized on CPU and would time the comparison out).
        if dtype_override is not None:
            wrapped = wrapped.to(dtype=dtype_override)
        return wrapped

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Random quantized latent for the 256x256 (0.06M) schedule:
        `z` of shape [B=1, embed_dim=16, 16, 16]."""
        arch = _ARCH[str(self._variant)]
        cd = arch["codebook_dim"]
        hw = arch["latent_hw"]
        gen = torch.Generator().manual_seed(0)
        z = torch.randn(1, cd, hw, hw, generator=gen)
        if dtype_override is not None:
            z = z.to(dtype=dtype_override)
        return {"z": z}
