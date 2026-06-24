# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FIBO VAE-decoder loader implementation.

FIBO (briaai/FIBO) uses an ``AutoencoderKLWan`` (the Wan 2.x video VAE, z_dim
48, spatial scale factor 16) hosted in the ``vae`` subfolder of the gated
``briaai/FIBO`` repo. In the text-to-image pipeline only the *decoder* is
exercised: denoised latents of shape ``[B, 48, 1, H/16, W/16]`` are decoded to
an ``[B, 3, 1, H, W]`` image. This loader wraps the VAE so its forward runs the
decode path at FIBO's native 1024x1024 resolution.

Reference: https://huggingface.co/briaai/FIBO
"""

from typing import Optional

import torch
from diffusers import AutoencoderKLWan

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


class _VaeDecodeWrapper(torch.nn.Module):
    """Expose ``AutoencoderKLWan.decode`` as the module forward.

    The diffusion pipeline only ever calls ``vae.decode(latents)``; wrapping it
    keeps the bringup graph to the decode path actually used at inference.
    """

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z, return_dict=False)[0]


class ModelVariant(StrEnum):
    """Available FIBO VAE variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """FIBO VAE-decoder (AutoencoderKLWan) loader."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="briaai/FIBO",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # FIBO's native output resolution (model card default) is 1024x1024, i.e. a
    # 64x64 latent. The AutoencoderKLWan decoder is extremely slow on host CPU
    # (the runner's CPU PCC reference would dominate the wall clock), so the
    # on-device component test decodes a smaller 256x256 (16x16 latent) tile;
    # the composite pipeline run decodes at the true native resolution.
    image_size = 256
    vae_scale_factor = 16
    z_dim = 48

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize the FIBO VAE loader.

        Args:
            variant: Optional ``ModelVariant`` — defaults to ``BASE``.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FIBO_vae",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FIBO VAE wrapped to its decode path.

        Args:
            dtype_override: Optional ``torch.dtype`` for the VAE weights.

        Returns:
            torch.nn.Module: ``_VaeDecodeWrapper`` around ``AutoencoderKLWan``.
        """
        model_kwargs = {"subfolder": "vae"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        vae = AutoencoderKLWan.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        vae.eval()
        if dtype_override is not None:
            vae = vae.to(dtype_override)
        return _VaeDecodeWrapper(vae)

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Return a sample latent tensor for the FIBO VAE decode.

        Args:
            dtype_override: Optional ``torch.dtype`` for the latent tensor.
            batch_size: Number of latents to batch.

        Returns:
            dict: ``z`` latent tensor of shape ``[B, 48, 1, H/16, W/16]``.
        """
        latent_dim = self.image_size // self.vae_scale_factor
        dtype = dtype_override if dtype_override is not None else torch.float32
        z = torch.randn(
            batch_size, self.z_dim, 1, latent_dim, latent_dim, dtype=dtype
        )
        return {"z": z}
