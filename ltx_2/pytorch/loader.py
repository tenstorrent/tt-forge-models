# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Lightricks LTX-2 audiovisual video-generation component loader.

LTX-2 (Lightricks/LTX-2) is a joint video + audio latent-diffusion pipeline
(diffusers LTX2Pipeline). It is brought up by independently-compilable
components, one ModelVariant each:

  - Transformer : LTX2VideoTransformer3DModel  (19B joint video+audio DiT, denoiser)
  - Vae         : AutoencoderKLLTX2Video        (3D causal video VAE, decoder path)
  - AudioVae    : AutoencoderKLLTX2Audio        (causal mel audio VAE, decoder path)
  - TextEncoder : Gemma3ForConditionalGeneration (Gemma-3 ~12B text encoder)

The scheduler, denoising loop, text connectors, vocoder and latent glue live in
host Python (the source LTX2Pipeline) and are not part of any compiled graph.
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
    LTX2_REPO_ID,
    MESH_NAMES,
    MESH_SHAPES,
    AudioVAEDecoderWrapper,
    GemmaTextEncoderWrapper,
    LTX2TransformerWrapper,
    VideoVAEDecoderWrapper,
    audio_vae_inputs,
    load_audio_vae,
    load_text_encoder,
    load_transformer,
    load_video_vae,
    text_encoder_inputs,
    transformer_inputs,
    video_vae_inputs,
)


class ModelVariant(StrEnum):
    """Independently loadable components of the LTX-2 pipeline."""

    TRANSFORMER = "Transformer"
    VAE = "Vae"
    AUDIO_VAE = "AudioVae"
    TEXT_ENCODER = "TextEncoder"


class ModelLoader(ForgeModel):
    """Load individual LTX-2 components without pulling the full pipeline.

    All components live in the Lightricks/LTX-2 repo and are loaded from their
    respective subfolders. The transformer (denoiser) is the compute-dominant
    component and the sharding target on multi-chip devices.
    """

    _VARIANTS = {
        ModelVariant.TRANSFORMER: ModelConfig(pretrained_model_name=LTX2_REPO_ID),
        ModelVariant.VAE: ModelConfig(pretrained_model_name=LTX2_REPO_ID),
        ModelVariant.AUDIO_VAE: ModelConfig(pretrained_model_name=LTX2_REPO_ID),
        ModelVariant.TEXT_ENCODER: ModelConfig(pretrained_model_name=LTX2_REPO_ID),
    }
    DEFAULT_VARIANT = ModelVariant.TRANSFORMER

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        task = (
            ModelTask.NLP_EMBED_GEN
            if variant == ModelVariant.TEXT_ENCODER
            else ModelTask.MM_VIDEO_TTT
        )
        return ModelInfo(
            model="LTX-2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the component for this variant as a torch.nn.Module.

        Returns:
            TRANSFORMER  -> LTX2TransformerWrapper (joint video+audio DiT)
            VAE          -> VideoVAEDecoderWrapper (decoder-only, plain tensor out)
            AUDIO_VAE    -> AudioVAEDecoderWrapper (decoder-only, plain tensor out)
            TEXT_ENCODER -> GemmaTextEncoderWrapper (packed Gemma-3 hidden states)
        """
        model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TRANSFORMER:
            return LTX2TransformerWrapper(load_transformer(model_name, dtype))
        if self._variant == ModelVariant.VAE:
            return VideoVAEDecoderWrapper(load_video_vae(model_name, dtype))
        if self._variant == ModelVariant.AUDIO_VAE:
            return AudioVAEDecoderWrapper(load_audio_vae(model_name, dtype))
        if self._variant == ModelVariant.TEXT_ENCODER:
            return GemmaTextEncoderWrapper(load_text_encoder(model_name, dtype))

        raise ValueError(f"Unknown variant: {self._variant}")

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Return a list of synthetic input tensors for the active component."""
        dtype = dtype_override if dtype_override is not None else DTYPE

        if self._variant == ModelVariant.TRANSFORMER:
            return transformer_inputs(dtype)
        if self._variant == ModelVariant.VAE:
            return video_vae_inputs(dtype)
        if self._variant == ModelVariant.AUDIO_VAE:
            return audio_vae_inputs(dtype)
        if self._variant == ModelVariant.TEXT_ENCODER:
            return text_encoder_inputs(dtype)

        raise ValueError(f"Unknown variant: {self._variant}")

    def get_mesh_config(self, num_devices: int):
        """Return (mesh_shape, mesh_names) for a ("batch", "model") 2D mesh.

        The VAEs fit comfortably on a single chip; the transformer (and text
        encoder) are the sharding targets on multi-chip devices.
        """
        if self._variant in (ModelVariant.VAE, ModelVariant.AUDIO_VAE):
            return (1, 1), MESH_NAMES

        if num_devices not in MESH_SHAPES:
            raise ValueError(
                f"Unsupported device count: {num_devices}. "
                f"Expected one of {sorted(MESH_SHAPES)}."
            )
        return MESH_SHAPES[num_devices], MESH_NAMES
