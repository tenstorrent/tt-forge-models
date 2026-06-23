# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 (Lightricks/LTX-2.3) 22B joint audio-video diffusion component loader.

LTX-2.3 is a single-model audio+video latent-diffusion pipeline. It is not yet
covered by ``diffusers`` and is distributed as one bundled ``.safetensors``
checkpoint with a ComfyUI-style key layout. This loader uses the official
``ltx-core`` reference package to build each independently-compilable component
from that bundled checkpoint:

  - Denoiser         -> LTXModel        (joint AV DiT)            ~18.99B params  [DEFAULT]
  - VideoVaeDecoder  -> VideoDecoder    (latents -> RGB frames)  ~0.41B params
  - AudioVaeDecoder  -> AudioDecoder    (latents -> mel spec)    ~0.03B params
  - Vocoder          -> VocoderWithBWE  (mel -> waveform)        ~0.13B params

The denoiser is the compute-dominant, gating component. The Gemma-3 text encoder
is external (not in the checkpoint) and is out of scope for this component set.
"""

from typing import Optional

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
from .src.model_utils import (
    DTYPE,
    LTX23_REPO,
    LTX23DenoiserWrapper,
    audio_vae_decoder_inputs,
    denoiser_inputs,
    load_audio_vae_decoder,
    load_denoiser,
    load_video_vae_decoder,
    load_vocoder,
    video_vae_decoder_inputs,
    vocoder_inputs,
)


class ModelVariant(StrEnum):
    """Independently loadable components of the LTX-2.3 pipeline."""

    DENOISER = "Denoiser"
    VIDEO_VAE_DECODER = "VideoVaeDecoder"
    AUDIO_VAE_DECODER = "AudioVaeDecoder"
    VOCODER = "Vocoder"


DEFAULT_VARIANT = ModelVariant.DENOISER


class ModelLoader(ForgeModel):
    """Load individual LTX-2.3 components from the bundled dev checkpoint.

    All variants resolve to the same ``Lightricks/LTX-2.3`` repo; the component
    is selected by the state-dict key filter applied inside ``ltx-core``'s
    single-file builder (see ``src/model_utils.py``).
    """

    _VARIANTS = {
        ModelVariant.DENOISER: ModelConfig(pretrained_model_name=LTX23_REPO),
        ModelVariant.VIDEO_VAE_DECODER: ModelConfig(pretrained_model_name=LTX23_REPO),
        ModelVariant.AUDIO_VAE_DECODER: ModelConfig(pretrained_model_name=LTX23_REPO),
        ModelVariant.VOCODER: ModelConfig(pretrained_model_name=LTX23_REPO),
    }
    DEFAULT_VARIANT = DEFAULT_VARIANT

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        # Denoiser / VAEs / vocoder are all part of the text+image+audio->video
        # generation pipeline; tag them as multimodal video text-to-thing.
        return ModelInfo(
            model="LTX-2.3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Build and return the component for this variant as a torch.nn.Module.

        Returns:
            DENOISER          -> LTX23DenoiserWrapper (plain-tensor forward over LTXModel)
            VIDEO_VAE_DECODER -> VideoDecoder
            AUDIO_VAE_DECODER -> AudioDecoder
            VOCODER           -> VocoderWithBWE
        """
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.DENOISER:
            return LTX23DenoiserWrapper(load_denoiser(dtype))
        if self._variant == ModelVariant.VIDEO_VAE_DECODER:
            return load_video_vae_decoder(dtype)
        if self._variant == ModelVariant.AUDIO_VAE_DECODER:
            return load_audio_vae_decoder(dtype)
        if self._variant == ModelVariant.VOCODER:
            return load_vocoder(dtype)

        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a dict of synthetic input tensors for the active component."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.DENOISER:
            return denoiser_inputs(dtype)
        if self._variant == ModelVariant.VIDEO_VAE_DECODER:
            return video_vae_decoder_inputs(dtype)
        if self._variant == ModelVariant.AUDIO_VAE_DECODER:
            return audio_vae_decoder_inputs(dtype)
        if self._variant == ModelVariant.VOCODER:
            return vocoder_inputs(dtype)

        raise ValueError(f"Unknown variant: {self._variant}")
