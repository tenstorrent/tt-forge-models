# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FIBO (briaai/FIBO) VAE-decoder component loader.

FIBO is BRIA AI's 8B-parameter DiT flow-matching text-to-image model. Its
output stage is the **Wan 2.2 VAE** (``AutoencoderKLWan`` in diffusers): after
the DiT denoising loop finishes, the per-channel-scaled latents are decoded
back to pixel space by ``vae.decode(...)``. This loader isolates that
**VAE decoder** as an independently-compilable component, following the
image-gen "one loader per component" convention (see the model-bringup skill's
``references/image-gen.md``). It is separate from ``fibo/pytorch`` (the DiT
denoiser) so the conv-heavy VAE can be brought up and tuned on its own.

Only the ``vae`` subfolder of ``briaai/FIBO`` is fetched (~2.8 GB) — the 8B
transformer and the SmolLM3-3B text encoder are never downloaded here.

The decoder is a 3D-convolutional network (``WanCausalConv3d`` stacks). It
consumes latents shaped ``[B, z_dim=48, T=1, H/16, W/16]`` and produces an
image ``[B, 3, T=1, H, W]``. At FIBO's native 1024x1024 resolution that is
``[1, 48, 1, 64, 64] -> [1, 3, 1, 1024, 1024]``.

Reference:
- Model card: https://huggingface.co/briaai/FIBO
- Decode call: diffusers ``pipeline_bria_fibo.BriaFiboPipeline.__call__`` —
  ``latents_scaled = latent / latents_std + latents_mean`` (per-channel), then
  ``image = self.vae.decode(latents_scaled, return_dict=False)[0]``.
"""

from typing import Optional

import torch

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

# FIBO's native generation resolution (model-card Generate example / pipeline
# defaults). The VAE spatial compression is 16x, so the latent grid is 64x64.
NATIVE_HEIGHT = 1024
NATIVE_WIDTH = 1024
VAE_SPATIAL_COMPRESSION = 16  # AutoencoderKLWan config: scale_factor_spatial
Z_DIM = 48  # AutoencoderKLWan config: z_dim (latent channels)


class ModelVariant(StrEnum):
    """Available FIBO VAE-decoder variants."""

    BASE = "Base"


class FiboVaeDecoderWrapper(torch.nn.Module):
    """Expose ``AutoencoderKLWan.decode`` as a plain positional ``forward``.

    The auto-runner (``DynamicTorchModelTester``) calls ``model(*inputs)`` and
    compares the returned tensor against a CPU golden. ``AutoencoderKLWan``'s
    own ``forward`` runs encode+decode; here we want just the decode path, so
    the wrapper's ``forward(latents)`` calls ``vae.decode`` and returns the
    reconstructed image tensor directly.
    """

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        out = self.vae.decode(latents, return_dict=False)[0]
        return out


class ModelLoader(ForgeModel):
    """FIBO VAE-decoder loader."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="briaai/FIBO",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize the loader for the given FIBO VAE-decoder variant.

        Args:
            variant: Optional ``ModelVariant`` — defaults to ``BASE``.
        """
        super().__init__(variant)
        self.vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FIBO-vae-decoder",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_vae(self, dtype_override=None):
        """Load (and cache) only the ``AutoencoderKLWan`` VAE from briaai/FIBO.

        Only the ``vae`` subfolder is fetched — the DiT transformer and text
        encoder are left untouched, keeping the download to ~2.8 GB.

        Note: ``briaai/FIBO`` is a gated repo (bria-fibo license). Accept the
        license on Hugging Face and authenticate via ``HF_TOKEN`` first.
        """
        if self.vae is not None:
            return self.vae

        from diffusers import AutoencoderKLWan

        kwargs = {"subfolder": "vae"}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        vae = AutoencoderKLWan.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False

        self.vae = vae
        return vae

    def load_model(self, dtype_override=None, **kwargs):
        """Return the wrapped FIBO VAE decoder.

        Args:
            dtype_override: Optional ``torch.dtype`` to cast the VAE to.

        Returns:
            torch.nn.Module: ``FiboVaeDecoderWrapper`` whose ``forward(latents)``
            runs ``vae.decode`` and returns the reconstructed image tensor.
        """
        vae = self._load_vae(dtype_override=dtype_override)
        if dtype_override is not None:
            vae = vae.to(dtype_override)
        return FiboVaeDecoderWrapper(vae)

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Return a representative scaled-latent input for the VAE decoder.

        The tensor matches what the FIBO pipeline feeds ``vae.decode`` at native
        1024x1024: shape ``[B, z_dim=48, T=1, 64, 64]``. To land in the latent
        distribution the VAE was trained to decode, a standard-normal draw is
        rescaled per-channel by the VAE's ``latents_std`` / ``latents_mean``
        (the inverse of the pipeline's ``latent / latents_std + latents_mean``
        normalization, where the config stores ``1/std``).

        Args:
            dtype_override: Optional ``torch.dtype`` for the latent tensor.
            batch_size: Batch size (default 1, matching the pipeline).

        Returns:
            tuple: ``(latents,)`` — a single positional tensor for the wrapper.
        """
        vae = self._load_vae(dtype_override=dtype_override)

        h_lat = NATIVE_HEIGHT // VAE_SPATIAL_COMPRESSION
        w_lat = NATIVE_WIDTH // VAE_SPATIAL_COMPRESSION

        # Deterministic latent so CPU golden and device run see identical input.
        generator = torch.Generator(device="cpu").manual_seed(42)
        raw = torch.randn(
            (batch_size, Z_DIM, 1, h_lat, w_lat),
            generator=generator,
            dtype=torch.float32,
        )

        # Rescale into the VAE's decode distribution (per-channel std/mean).
        mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(
            1, Z_DIM, 1, 1, 1
        )
        std = torch.tensor(vae.config.latents_std, dtype=torch.float32).view(
            1, Z_DIM, 1, 1, 1
        )
        latents = raw * std + mean

        if dtype_override is not None:
            latents = latents.to(dtype_override)
        return (latents,)
