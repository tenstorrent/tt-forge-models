# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MS-LC-EQ-D-VR VAE model loader implementation.

Loads Anzhc/MS-LC-EQ-D-VR_VAE, a family of EQ-VAE finetunes based on the
SDXL and FLUX autoencoders. Each variant is distributed as a single
safetensors checkpoint on the hub and is loaded through
AutoencoderKL.from_single_file using the matching base VAE config.

Available variants (SDXL-based, 4 latent channels):
- BASE: First version
- B2, B3, B4, B5: Progressive SDXL finetunes (B5 is the latest)
- EQB7: EQB7 checkpoint
- PAD_EQB7_DEC_B2: Padded EQB7 with B2 decoder
- PAD_EQB7_DECODER_ONLY: Padded EQB7 decoder-only pass
- FP32: fp32 weights version

Available variants (FLUX-based, 16 latent channels):
- FLUX: FLUX VAE finetune
- PAD_FLUX_EQ_V2_B1: Padded FLUX EQ v2 B1
"""

from typing import Any, Optional

import torch
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_download

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

REPO_ID = "Anzhc/MS-LC-EQ-D-VR_VAE"

# Base VAE configs used to instantiate the AutoencoderKL architecture.
# The FLUX mirror is used instead of black-forest-labs/FLUX.1-dev because
# the upstream repo is gated and cannot be fetched without a token.
SDXL_VAE_CONFIG = "stabilityai/sdxl-vae"
FLUX_VAE_CONFIG_REPO = "camenduru/FLUX.1-dev-ungated"
FLUX_VAE_CONFIG_SUBFOLDER = "vae"

# Latent dimensions
SDXL_LATENT_CHANNELS = 4
FLUX_LATENT_CHANNELS = 16
LATENT_HEIGHT = 8
LATENT_WIDTH = 8


class ModelVariant(StrEnum):
    """Available MS-LC-EQ-D-VR VAE model variants."""

    BASE = "base"
    B2 = "b2"
    B3 = "b3"
    B4 = "b4"
    B5 = "b5"
    EQB7 = "eqb7"
    PAD_EQB7_DEC_B2 = "pad_eqb7_dec_b2"
    PAD_EQB7_DECODER_ONLY = "pad_eqb7_decoder_only"
    FP32 = "fp32"
    FLUX = "flux"
    PAD_FLUX_EQ_V2_B1 = "pad_flux_eq_v2_b1"


# Mapping from variant -> safetensors filename inside the HF repo
_VARIANT_FILENAMES = {
    ModelVariant.BASE: "MS-LC-EQ-D-VR VAE.safetensors",
    ModelVariant.B2: "MS-LC-EQ-D-VR VAE B2.safetensors",
    ModelVariant.B3: "MS-LC-EQ-D-VR VAE B3.safetensors",
    ModelVariant.B4: "MS-LC-EQ-D-VR VAE B4.safetensors",
    ModelVariant.B5: "MS-LC-EQ-D-VR VAE B5.safetensors",
    ModelVariant.EQB7: "EQB7.safetensors",
    ModelVariant.PAD_EQB7_DEC_B2: "Pad EQB7 DecB2.safetensors",
    ModelVariant.PAD_EQB7_DECODER_ONLY: "Pad EQB7 Decoder-only-pass.safetensors",
    ModelVariant.FP32: "MS-LC-EQ-D-VR VAE fp32 weights.safetensors",
    ModelVariant.FLUX: "MS-LC-EQ-D-VR VAE FLUX.safetensors",
    ModelVariant.PAD_FLUX_EQ_V2_B1: "Pad Flux EQ v2 B1.safetensors",
}

_FLUX_VARIANTS = {ModelVariant.FLUX, ModelVariant.PAD_FLUX_EQ_V2_B1}


class ModelLoader(ForgeModel):
    """MS-LC-EQ-D-VR VAE model loader."""

    _VARIANTS = {
        variant: ModelConfig(pretrained_model_name=REPO_ID) for variant in ModelVariant
    }
    DEFAULT_VARIANT = ModelVariant.B5

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MS_LC_EQ_D_VR_VAE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _is_flux(self) -> bool:
        return self._variant in _FLUX_VARIANTS

    def _latent_channels(self) -> int:
        return FLUX_LATENT_CHANNELS if self._is_flux() else SDXL_LATENT_CHANNELS

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the AutoencoderKL VAE model for the selected variant."""
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._vae is None:
            filename = _VARIANT_FILENAMES[self._variant]
            weights_path = hf_hub_download(repo_id=REPO_ID, filename=filename)

            if self._is_flux():
                self._vae = AutoencoderKL.from_single_file(
                    weights_path,
                    config=FLUX_VAE_CONFIG_REPO,
                    subfolder=FLUX_VAE_CONFIG_SUBFOLDER,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=False,
                )
            else:
                self._vae = AutoencoderKL.from_single_file(
                    weights_path,
                    config=SDXL_VAE_CONFIG,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=False,
                )
            self._vae = self._vae.to(dtype=dtype)
            self._vae.eval()
        elif dtype_override is not None:
            self._vae = self._vae.to(dtype=dtype_override)
        return self._vae

    def load_inputs(
        self, dtype_override: Optional[torch.dtype] = None, **kwargs
    ) -> Any:
        """Prepare inputs for the VAE.

        Pass vae_type="decoder" (default) or vae_type="encoder".
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        vae_type = kwargs.get("vae_type", "decoder")
        latent_channels = self._latent_channels()

        if vae_type == "decoder":
            return torch.randn(
                1,
                latent_channels,
                LATENT_HEIGHT,
                LATENT_WIDTH,
                dtype=dtype,
            )
        elif vae_type == "encoder":
            return torch.randn(1, 3, LATENT_HEIGHT * 8, LATENT_WIDTH * 8, dtype=dtype)
        else:
            raise ValueError(
                f"Unknown vae_type: {vae_type}. Expected 'decoder' or 'encoder'."
            )
