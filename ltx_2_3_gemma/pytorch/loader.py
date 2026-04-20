# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 Gemma model loader for tt_forge_models.

LTX-2.3 Gemma is a DiT-based (Diffusion Transformer) audio-video foundation model
by Lightricks, bundled with a Gemma 3 12B vision-language model for enhanced
text/image understanding. It generates synchronized video and audio.

Repository: https://huggingface.co/lightweight/LTX-2.3_Gemma

Available subfolders:
- transformer: LTX2VideoTransformer3DModel
- vae: AutoencoderKLLTX2Video
- audio_vae: AutoencoderKLLTX2Audio
"""

import logging
from typing import Any, Optional

import torch
from diffusers import LTX2Pipeline
from diffusers.models import LTX2VideoTransformer3DModel

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

logger = logging.getLogger(__name__)

FALLBACK_MODEL_NAME = "Lightricks/LTX-2"
SUPPORTED_SUBFOLDERS = {"transformer", "vae", "audio_vae"}

VAE_SPATIAL_COMPRESSION_RATIO = 32
VAE_TEMPORAL_COMPRESSION_RATIO = 8
AUDIO_VAE_MEL_COMPRESSION_RATIO = 4
AUDIO_VAE_TEMPORAL_COMPRESSION_RATIO = 4
AUDIO_SAMPLING_RATE = 16000
AUDIO_HOP_LENGTH = 160
AUDIO_VAE_MEL_BINS = 64


class ModelVariant(StrEnum):
    """Available LTX-2.3 Gemma variants."""

    DEFAULT = "default"


class ModelLoader(ForgeModel):
    """
    Loader for LTX-2.3 Gemma audio-video generation model.

    Supports loading the full pipeline or individual components via subfolder:
    - 'transformer': LTX2VideoTransformer3DModel
    - 'vae': AutoencoderKLLTX2Video
    - 'audio_vae': AutoencoderKLLTX2Audio
    """

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="lightweight/LTX-2.3_Gemma",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

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
        self.pipeline: Optional[LTX2Pipeline] = None
        self._transformer: Optional[LTX2VideoTransformer3DModel] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LTX-2.3 Gemma",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer_from_config(
        self, dtype: torch.dtype
    ) -> LTX2VideoTransformer3DModel:
        for repo in [
            self._variant_config.pretrained_model_name,
            FALLBACK_MODEL_NAME,
        ]:
            try:
                config = LTX2VideoTransformer3DModel.load_config(
                    repo, subfolder="transformer"
                )
                self._transformer = LTX2VideoTransformer3DModel.from_config(config)
                self._transformer = self._transformer.to(dtype)
                logger.info("Loaded transformer config from %s", repo)
                return self._transformer
            except Exception:
                logger.warning(
                    "Cannot load transformer config from %s, trying next", repo
                )
        raise RuntimeError("Failed to load transformer config from any source")

    def _load_pipeline(self, dtype: torch.dtype) -> LTX2Pipeline:
        try:
            self.pipeline = LTX2Pipeline.from_pretrained(
                self._variant_config.pretrained_model_name,
                torch_dtype=dtype,
            )
        except Exception:
            logger.warning(
                "Gated repo %s is not accessible, falling back to %s",
                self._variant_config.pretrained_model_name,
                FALLBACK_MODEL_NAME,
            )
            self.pipeline = LTX2Pipeline.from_pretrained(
                FALLBACK_MODEL_NAME,
                torch_dtype=dtype,
            )
        return self.pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._subfolder == "vae" or self._subfolder == "audio_vae":
            if self.pipeline is None:
                self._load_pipeline(dtype)
            if self._subfolder == "vae":
                return self.pipeline.vae
            return self.pipeline.audio_vae

        if self._transformer is None:
            self._load_transformer_from_config(dtype)
        return self._transformer

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._subfolder == "transformer" or self._subfolder is None:
            if self._transformer is None:
                self._load_transformer_from_config(dtype)
            return self._load_transformer_inputs(dtype)

        if self.pipeline is None:
            self._load_pipeline(dtype)

        if self._subfolder == "vae":
            vae_type = kwargs.get("vae_type", "decoder")
            if vae_type == "decoder":
                return self._load_vae_decoder_inputs(dtype)
            else:
                return self._load_vae_encoder_inputs(dtype)
        elif self._subfolder == "audio_vae":
            vae_type = kwargs.get("vae_type", "decoder")
            if vae_type == "decoder":
                return self._load_audio_vae_decoder_inputs(dtype)
            else:
                return self._load_audio_vae_encoder_inputs(dtype)

    def _load_transformer_inputs(self, dtype: torch.dtype) -> dict:
        batch_size = 1
        height = 64
        width = 64
        num_frames = 9

        latent_height = height // VAE_SPATIAL_COMPRESSION_RATIO
        latent_width = width // VAE_SPATIAL_COMPRESSION_RATIO
        latent_num_frames = (num_frames - 1) // VAE_TEMPORAL_COMPRESSION_RATIO + 1

        in_channels = self._transformer.config.in_channels
        video_seq_len = latent_num_frames * latent_height * latent_width
        hidden_states = torch.randn(batch_size, video_seq_len, in_channels, dtype=dtype)

        frame_rate = 24.0
        duration_s = num_frames / frame_rate
        audio_latents_per_second = (
            AUDIO_SAMPLING_RATE
            / AUDIO_HOP_LENGTH
            / float(AUDIO_VAE_TEMPORAL_COMPRESSION_RATIO)
        )
        audio_num_frames = max(1, round(duration_s * audio_latents_per_second))
        latent_mel_bins = AUDIO_VAE_MEL_BINS // AUDIO_VAE_MEL_COMPRESSION_RATIO
        audio_in_channels = self._transformer.config.audio_in_channels
        audio_hidden_states = torch.randn(
            batch_size,
            audio_num_frames,
            audio_in_channels * latent_mel_bins,
            dtype=dtype,
        )

        max_seq_len = 64
        caption_channels = self._transformer.config.caption_channels
        encoder_hidden_states = torch.randn(
            batch_size, max_seq_len, caption_channels, dtype=dtype
        )
        audio_encoder_hidden_states = torch.randn(
            batch_size, max_seq_len, caption_channels, dtype=dtype
        )

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "audio_hidden_states": audio_hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "audio_encoder_hidden_states": audio_encoder_hidden_states,
            "timestep": timestep,
            "num_frames": latent_num_frames,
            "height": latent_height,
            "width": latent_width,
            "fps": frame_rate,
            "audio_num_frames": audio_num_frames,
            "return_dict": False,
        }

    def _load_vae_decoder_inputs(self, dtype: torch.dtype) -> dict:
        latent_channels = self.pipeline.vae.config.latent_channels
        return {
            "sample": torch.randn(1, latent_channels, 2, 2, 2, dtype=dtype),
        }

    def _load_vae_encoder_inputs(self, dtype: torch.dtype) -> dict:
        return {
            "sample": torch.randn(1, 3, 9, 64, 64, dtype=dtype),
        }

    def _load_audio_vae_decoder_inputs(self, dtype: torch.dtype) -> dict:
        latent_channels = self.pipeline.audio_vae.config.latent_channels
        mel_bins = self.pipeline.audio_vae.config.mel_bins
        audio_vae_mel = self.pipeline.audio_vae_mel_compression_ratio
        latent_mel_bins = mel_bins // audio_vae_mel
        return {
            "sample": torch.randn(1, latent_channels, 4, latent_mel_bins, dtype=dtype),
        }

    def _load_audio_vae_encoder_inputs(self, dtype: torch.dtype) -> dict:
        mel_bins = self.pipeline.audio_vae.config.mel_bins
        return {
            "sample": torch.randn(1, 1, 16, mel_bins, dtype=dtype),
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
