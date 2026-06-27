# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image VAE decoder loader for text-to-image generation.

Brings up the decode path of ``AutoencoderKLQwenImage`` — the component that
turns the denoised latents into the final RGB image. A thin wrapper exposes
``decode`` as ``forward`` so the generic single-forward-pass test harness
exercises the decoder.
"""
import torch
from typing import Optional

from diffusers import AutoencoderKLQwenImage

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


class ModelVariant(StrEnum):
    """Available Qwen-Image VAE variants."""

    BASE = "Base"


class _QwenImageVAEDecoder(torch.nn.Module):
    """Wraps AutoencoderKLQwenImage so forward() runs the decoder only.

    The stock ``decode`` path calls ``clear_cache()`` -> ``_count_conv3d`` (a
    nested helper iterating ``model.modules()``) on every call, which breaks
    under torch.compile/dynamo. We pre-count the temporal-cache conv slots once
    eagerly (in the loader) and replicate the per-frame decode here without the
    module-walking helper, so the compiled graph only contains the actual
    post-quant conv + decoder ops.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        vae = self.vae
        num_frame = latents.shape[2]
        # Reset the temporal feature cache (slot count precomputed in loader).
        vae._conv_idx = [0]
        vae._feat_map = [None] * vae._conv_num
        x = vae.post_quant_conv(latents)
        out = None
        for i in range(num_frame):
            vae._conv_idx = [0]
            frame = vae.decoder(
                x[:, :, i : i + 1, :, :],
                feat_cache=vae._feat_map,
                feat_idx=vae._conv_idx,
            )
            out = frame if out is None else torch.cat([out, frame], dim=2)
        return torch.clamp(out, min=-1.0, max=1.0)


class ModelLoader(ForgeModel):
    """Qwen-Image VAE decoder loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="Qwen/Qwen-Image",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # Native generation resolution (pipeline default — 1024x1024).
    HEIGHT = 1024
    WIDTH = 1024

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.vae = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="qwen_image_vae",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Qwen-Image VAE decoder wrapper.

        Args:
            dtype_override: Optional torch.dtype to override the model's dtype.

        Returns:
            torch.nn.Module: Decoder wrapper around AutoencoderKLQwenImage.
        """
        model_kwargs = {"subfolder": "vae", "use_safetensors": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        vae = AutoencoderKLQwenImage.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        vae = vae.eval()
        if dtype_override is not None:
            vae = vae.to(dtype_override)

        # Pre-count temporal-cache conv slots once (eager) so the compiled
        # decode path never has to walk model.modules() (dynamo-incompatible).
        vae.clear_cache()

        self.vae = vae
        return _QwenImageVAEDecoder(vae)

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample latents for the Qwen-Image VAE decoder at native resolution.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: {"latents": tensor [B, z_dim, T, H/8, W/8]}.
        """
        if self.vae is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        z_dim = self.vae.config.z_dim  # 16
        # AutoencoderKLQwenImage spatially downsamples by 8; single image => 1 frame.
        latent_h = self.HEIGHT // 8  # 128
        latent_w = self.WIDTH // 8  # 128
        num_frames = 1

        latents = torch.randn(
            batch_size, z_dim, num_frames, latent_h, latent_w, dtype=dtype
        )

        return {"latents": latents}
