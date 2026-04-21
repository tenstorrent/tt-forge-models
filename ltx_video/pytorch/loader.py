# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-Video model loader for tt_forge_models.

LTX-Video is a text-to-video diffusion model using a 3D transformer backbone
with a T5 text encoder and a video VAE. The spatial upscaler variant swaps the
transformer for a spatial latent upsampler that doubles the height/width of
LTX-Video latents.

Repositories:
- https://huggingface.co/optimum-intel-internal-testing/tiny-random-ltx-video
- https://huggingface.co/Lightricks/LTX-Video-0.9.8-13B-distilled
- https://huggingface.co/Lightricks/ltxv-spatial-upscaler-0.9.7

Available subfolders:
- transformer: LTXVideoTransformer3DModel
- vae: AutoencoderKLLTXVideo
- latent_upsampler: LTXLatentUpsamplerModel (spatial upscaler variants only)
"""

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline

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

SUPPORTED_SUBFOLDERS = {"transformer", "vae", "latent_upsampler"}


class ModelVariant(StrEnum):
    """Available LTX-Video variants."""

    TINY_RANDOM = "tiny_random"
    LTX_VIDEO_0_9_7 = "LTX_Video_0_9_7"
    LTX_VIDEO_0_9_8_13B_DISTILLED = "LTX_Video_0_9_8_13B_distilled"
    LTXV_SPATIAL_UPSCALER_0_9_7 = "LTXV_Spatial_Upscaler_0_9_7"


SPATIAL_UPSCALER_VARIANTS = {ModelVariant.LTXV_SPATIAL_UPSCALER_0_9_7}


class ModelLoader(ForgeModel):
    """
    Loader for LTX-Video text-to-video generation model.

    Supports loading the full pipeline or individual components via subfolder:
    - 'transformer': LTXVideoTransformer3DModel
    - 'vae': AutoencoderKLLTXVideo
    - 'latent_upsampler': LTXLatentUpsamplerModel (spatial upscaler variants)
    """

    _VARIANTS = {
        ModelVariant.TINY_RANDOM: ModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-random-ltx-video",
        ),
        ModelVariant.LTX_VIDEO_0_9_7: ModelConfig(
            pretrained_model_name="a-r-r-o-w/LTX-Video-0.9.7-diffusers",
        ),
        ModelVariant.LTX_VIDEO_0_9_8_13B_DISTILLED: ModelConfig(
            pretrained_model_name="Lightricks/LTX-Video-0.9.8-13B-distilled",
        ),
        ModelVariant.LTXV_SPATIAL_UPSCALER_0_9_7: ModelConfig(
            pretrained_model_name="Lightricks/ltxv-spatial-upscaler-0.9.7",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM

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
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LTX-Video",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype) -> DiffusionPipeline:
        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )
        return self.pipeline

    def _default_subfolder(self) -> str:
        if self._variant in SPATIAL_UPSCALER_VARIANTS:
            return "latent_upsampler"
        return "transformer"

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self.pipeline is None:
            self._load_pipeline(dtype)

        subfolder = self._subfolder or self._default_subfolder()

        if subfolder == "vae":
            return self.pipeline.vae
        elif subfolder == "latent_upsampler":
            return self.pipeline.latent_upsampler
        else:
            return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.float32

        if self.pipeline is None:
            self._load_pipeline(dtype)

        subfolder = self._subfolder or self._default_subfolder()

        if subfolder == "vae":
            vae_type = kwargs.get("vae_type", "decoder")
            if vae_type == "decoder":
                return self._load_vae_decoder_inputs(dtype)
            else:
                return self._load_vae_encoder_inputs(dtype)
        elif subfolder == "latent_upsampler":
            return self._load_latent_upsampler_inputs(dtype)
        else:
            return self._load_transformer_inputs(dtype)

    def _load_transformer_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic inputs for the LTXVideo transformer forward pass."""
        batch_size = 1
        config = self.pipeline.transformer.config

        latent_num_frames = 2
        latent_height = 2
        latent_width = 2
        video_seq_len = latent_num_frames * latent_height * latent_width

        hidden_states = torch.randn(
            batch_size, video_seq_len, config.in_channels, dtype=dtype
        )

        caption_channels = config.caption_channels
        encoder_hidden_states = torch.randn(
            batch_size, 8, caption_channels, dtype=dtype
        )
        encoder_attention_mask = torch.ones(batch_size, 8, dtype=dtype)

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "timestep": timestep,
            "num_frames": latent_num_frames,
            "height": latent_height,
            "width": latent_width,
            "return_dict": False,
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

    def _load_latent_upsampler_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic inputs for the LTX latent upsampler."""
        in_channels = self.pipeline.latent_upsampler.config.in_channels
        return {
            "hidden_states": torch.randn(1, in_channels, 2, 8, 8, dtype=dtype),
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
