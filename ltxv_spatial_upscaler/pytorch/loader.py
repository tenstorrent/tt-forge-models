# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-Video Spatial Upscaler model loader for tt_forge_models.

LTX-Video Spatial Upscaler 0.9.7 is a diffusion-based latent video spatial
upscaler by Lightricks. It enhances the spatial resolution of LTX-Video
latents by 2x in height and width, typically used as a stage in a multi-stage
LTX-Video generation pipeline.

Repositories:
- https://huggingface.co/Lightricks/ltxv-spatial-upscaler-0.9.7

Available subfolders:
- latent_upsampler: LTXLatentUpsamplerModel (default)
- vae: AutoencoderKLLTXVideo
"""

from typing import Any, Optional

import torch
from diffusers import LTXLatentUpsamplePipeline

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

SUPPORTED_SUBFOLDERS = {"latent_upsampler", "vae"}


class ModelVariant(StrEnum):
    """Available LTX-Video Spatial Upscaler variants."""

    LTXV_SPATIAL_UPSCALER_0_9_7 = "0.9.7"


class ModelLoader(ForgeModel):
    """
    Loader for the LTX-Video Spatial Upscaler pipeline.

    Supports loading the full pipeline or individual components via subfolder:
    - 'latent_upsampler': LTXLatentUpsamplerModel (default)
    - 'vae': AutoencoderKLLTXVideo
    """

    _VARIANTS = {
        ModelVariant.LTXV_SPATIAL_UPSCALER_0_9_7: ModelConfig(
            pretrained_model_name="Lightricks/ltxv-spatial-upscaler-0.9.7",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LTXV_SPATIAL_UPSCALER_0_9_7

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        subfolder: Optional[str] = None,
    ):
        super().__init__(variant)
        if subfolder is not None and subfolder not in SUPPORTED_SUBFOLDERS:
            raise ValueError(
                f"Unknown subfolder: {subfolder}. Supported: {SUPPORTED_SUBFOLDERS}"
            )
        self._subfolder = subfolder
        self.pipeline: Optional[LTXLatentUpsamplePipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LTX-Video Spatial Upscaler",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype) -> LTXLatentUpsamplePipeline:
        self.pipeline = LTXLatentUpsamplePipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )
        return self.pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipeline is None:
            self._load_pipeline(dtype)

        if self._subfolder == "vae":
            return self.pipeline.vae
        return self.pipeline.latent_upsampler

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.pipeline is None:
            self._load_pipeline(dtype)

        if self._subfolder == "vae":
            vae_type = kwargs.get("vae_type", "decoder")
            if vae_type == "decoder":
                return self._load_vae_decoder_inputs(dtype)
            return self._load_vae_encoder_inputs(dtype)
        return self._load_latent_upsampler_inputs(dtype)

    def _load_latent_upsampler_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic inputs for the latent spatial upsampler."""
        latent_channels = self.pipeline.vae.config.latent_channels
        # Video latents: [B, C, F, H, W]
        return {
            "hidden_states": torch.randn(1, latent_channels, 2, 8, 8, dtype=dtype),
        }

    def _load_vae_decoder_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic decoder inputs for the video VAE."""
        latent_channels = self.pipeline.vae.config.latent_channels
        return {
            "sample": torch.randn(1, latent_channels, 2, 2, 2, dtype=dtype),
        }

    def _load_vae_encoder_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic encoder inputs for the video VAE."""
        return {
            "sample": torch.randn(1, 3, 9, 64, 64, dtype=dtype),
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
