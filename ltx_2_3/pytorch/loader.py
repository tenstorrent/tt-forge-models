# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
LTX-2.3 (Lightricks) model loader for tt_forge_models.

LTX-2.3 is a 22B DiT-based *joint audio-video* foundation model (text/image -> video+audio).
Repository: https://huggingface.co/Lightricks/LTX-2.3

It is NOT yet supported by `diffusers` (model card: "Diffusers support coming soon");
it ships as one LTX-native 46 GB `.safetensors` checkpoint that packs the DiT
denoiser, video VAE, audio VAE and vocoder together (extracted by key-prefix
filters). Components are built via `ltx_core` (`pip install ltx-core`). The
Gemma-3-12B text encoder lives in a separate gated repo and is not bundled here.

This loader exposes the independently-compilable components via `subfolder`:
    - "video_decoder": video VAE decoder  (default — the most tractable to bring up)
    - "video_encoder": video VAE encoder
    - "transformer":   the 22B DiT denoiser (X0Model; joint audio-video Modality forward)
    - "audio_decoder": audio VAE decoder
    - "vocoder":       HiFi-GAN vocoder
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch

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
from .src import utils

SUPPORTED_SUBFOLDERS = {
    "video_decoder",
    "video_encoder",
    "transformer",
    "audio_decoder",
    "vocoder",
}


@dataclass
class LTXConfig(ModelConfig):
    """Configuration for LTX-2.3 variants."""

    source: ModelSource = ModelSource.HUGGING_FACE
    checkpoint_file: str = utils.DEFAULT_CHECKPOINT


class ModelVariant(StrEnum):
    """Available LTX-2.3 variants."""

    DEV_22B = "22b-dev"
    DISTILLED_22B = "22b-distilled"


class ModelLoader(ForgeModel):
    """Loader for the LTX-2.3 audio-video generation model (per-component)."""

    _VARIANTS = {
        ModelVariant.DEV_22B: LTXConfig(
            pretrained_model_name="Lightricks/LTX-2.3",
            checkpoint_file="ltx-2.3-22b-dev.safetensors",
        ),
        ModelVariant.DISTILLED_22B: LTXConfig(
            pretrained_model_name="Lightricks/LTX-2.3",
            checkpoint_file="ltx-2.3-22b-distilled.safetensors",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEV_22B

    def __init__(
        self, variant: Optional[ModelVariant] = None, subfolder: Optional[str] = None
    ):
        """
        Args:
            variant: Model variant to load.
            subfolder: Which component to load (see SUPPORTED_SUBFOLDERS).
                Defaults to "video_decoder" — the most tractable single-forward
                component to validate on device.
        """
        super().__init__(variant)
        if subfolder is None:
            subfolder = "video_decoder"
        if subfolder not in SUPPORTED_SUBFOLDERS:
            raise ValueError(
                f"Unknown subfolder: {subfolder}. Supported: {SUPPORTED_SUBFOLDERS}"
            )
        self._subfolder = subfolder
        self._checkpoint_path = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LTX-2.3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_VIDEO_TTT,  # Video generation
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _ensure_checkpoint(self) -> str:
        if self._checkpoint_path is None:
            config = self._variant_config
            self._checkpoint_path = utils.download_checkpoint(
                config.pretrained_model_name, config.checkpoint_file
            )
        return self._checkpoint_path

    def load_model(self, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        ckpt = self._ensure_checkpoint()

        if self._subfolder == "video_decoder":
            model = utils.load_video_decoder(ckpt, dtype)
            # Deterministic forward for CPU-vs-TT comparison.
            utils.disable_vae_noise(model)
            return model
        elif self._subfolder == "video_encoder":
            return utils.load_video_encoder(ckpt, dtype)
        elif self._subfolder == "transformer":
            return utils.load_transformer(ckpt, dtype)
        elif self._subfolder == "audio_decoder":
            return utils.load_audio_decoder(ckpt, dtype)
        elif self._subfolder == "vocoder":
            return utils.load_vocoder(ckpt, dtype)
        else:
            raise ValueError(f"Unknown subfolder: {self._subfolder}")

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._subfolder == "video_decoder":
            # Reduced clip by default so the first device compile of the conv3d
            # decoder is tractable; pass latent_f/h/w for native resolution.
            latent_f = kwargs.get("latent_f", 2)
            latent_h = kwargs.get("latent_h", 8)
            latent_w = kwargs.get("latent_w", 8)
            return {
                "sample": utils.video_decoder_latent(dtype, latent_f, latent_h, latent_w)
            }
        elif self._subfolder == "video_encoder":
            frames = kwargs.get("frames", 9)
            height = kwargs.get("height", 256)
            width = kwargs.get("width", 256)
            return {"sample": utils.video_encoder_pixels(dtype, frames, height, width)}
        elif self._subfolder == "transformer":
            return utils.transformer_video_inputs(
                dtype,
                kwargs.get("latent_f", 2),
                kwargs.get("latent_h", 8),
                kwargs.get("latent_w", 8),
            )
        else:
            raise RuntimeError(
                f"load_inputs not implemented for subfolder={self._subfolder!r}. "
                "The transformer/audio components use non-tensor (Modality / spectrogram) "
                "inputs; see src/utils.py."
            )

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        """Unpack a component's forward output to a single tensor."""
        if hasattr(output, "sample"):
            return output.sample
        if isinstance(output, (tuple, list)):
            # transformer returns (video, audio); take the video stream
            return output[0]
        return output
