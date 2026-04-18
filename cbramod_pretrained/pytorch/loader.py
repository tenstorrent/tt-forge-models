# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CBraMod pre-trained EEG foundation model loader implementation.
"""

import math

import torch
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


def _build_dft_matrices(n):
    """Build cos/sin DFT matrices for rfft with norm='forward'."""
    k = torch.arange(n // 2 + 1).unsqueeze(0).float()
    t = torch.arange(n).unsqueeze(1).float()
    angle = 2.0 * math.pi * t * k / n
    cos_mat = torch.cos(angle) / n
    sin_mat = torch.sin(angle) / n
    return cos_mat, sin_mat


def _patch_embedding_forward_no_fft(self, x, mask=None):
    """Replacement forward for _PatchEmbedding that avoids torch.fft.rfft."""
    from einops import rearrange

    bz, ch_num, patch_num, patch_size = x.shape
    if mask is None:
        mask_x = x
    else:
        mask_x = x.clone()
        mask_x[mask == 1] = self.mask_encoding

    mask_x = rearrange(mask_x, "b c n p -> b 1 (c n) p")
    patch_emb = self.proj_in(mask_x)
    patch_emb = rearrange(patch_emb, "b d (c n) p2 -> b c n (d p2)", c=ch_num)

    mask_x = rearrange(mask_x, "b 1 (c n) p -> (b c n) p", c=ch_num)
    real_part = mask_x @ self._dft_cos.to(mask_x.device, mask_x.dtype)
    imag_part = mask_x @ self._dft_sin.to(mask_x.device, mask_x.dtype)
    spectral = torch.sqrt(real_part * real_part + imag_part * imag_part)
    spectral = rearrange(
        spectral,
        "(b c n) p -> b c n p",
        b=bz,
        c=ch_num,
        p=patch_size // 2 + 1,
    )
    spectral_emb = self.spectral_proj(spectral)

    patch_emb = patch_emb + spectral_emb

    positional_embedding = self.positional_encoding(
        rearrange(patch_emb, "b c n p -> b p c n", p=self.d_model)
    )
    positional_embedding = rearrange(positional_embedding, "b p c n -> b c n p")

    patch_emb = patch_emb + positional_embedding

    return patch_emb


def _replace_fft_with_matmul(model):
    """Replace FFT in PatchEmbedding with DFT matrix multiplication."""
    patch_emb = model.patch_embedding
    patch_size = patch_emb.patch_size
    cos_mat, sin_mat = _build_dft_matrices(patch_size)
    patch_emb.register_buffer("_dft_cos", cos_mat)
    patch_emb.register_buffer("_dft_sin", sin_mat)
    import types

    patch_emb.forward = types.MethodType(_patch_embedding_forward_no_fft, patch_emb)
    return model


class ModelVariant(StrEnum):
    """Available CBraMod model variants."""

    PRETRAINED = "Pretrained"


class ModelLoader(ForgeModel):
    """CBraMod pre-trained EEG foundation model loader.

    CBraMod (Criss-Cross Brain Foundation Model) is a foundation model
    for EEG decoding using criss-cross spatial and temporal attention.
    Pre-trained on the Temple University Hospital EEG Corpus (TUEG).
    """

    _VARIANTS = {
        ModelVariant.PRETRAINED: ModelConfig(
            pretrained_model_name="braindecode/cbramod-pretrained",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PRETRAINED

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="CBraMod",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from braindecode.models import CBraMod

        model = CBraMod.from_pretrained(
            self._variant_config.pretrained_model_name,
            return_encoder_output=True,
        )

        _replace_fft_with_matmul(model)

        model.eval()

        if dtype_override is not None and dtype_override != torch.bfloat16:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        if dtype_override == torch.bfloat16:
            dtype_override = None
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)

        n_channels = 16
        n_times = 2000
        inputs = torch.randn(1, n_channels, n_times, dtype=dtype)

        return inputs
