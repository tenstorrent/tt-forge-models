# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AudioLDM model loader implementation for text-to-audio generation
"""
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AudioLDMPipeline
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available AudioLDM model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """AudioLDM model loader implementation for text-to-audio generation tasks."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="cvssp/audioldm",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="AudioLDM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load the AudioLDM pipeline."""
        pipe_kwargs = {}
        if dtype_override is not None:
            pipe_kwargs["torch_dtype"] = dtype_override

        self.pipe = AudioLDMPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, **pipe_kwargs
        )

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the AudioLDM UNet model."""
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipe.unet = self.pipe.unet.to(dtype_override)

        return self.pipe.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the AudioLDM UNet model."""
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32
        prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
        audio_length_in_s = 5.0

        # Encode text prompt through CLAP text encoder (returns L2-normalized
        # pooled text embeddings used as class_labels for the UNet).
        text_inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = self.pipe.text_encoder(
            text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
        ).text_embeds
        prompt_embeds = F.normalize(prompt_embeds, dim=-1).to(dtype=dtype)

        # Compute mel-spectrogram latent dimensions following the pipeline.
        vocoder_upsample_factor = (
            np.prod(self.pipe.vocoder.config.upsample_rates)
            / self.pipe.vocoder.config.sampling_rate
        )
        height = int(audio_length_in_s / vocoder_upsample_factor)
        vae_scale_factor = self.pipe.vae_scale_factor
        model_in_dim = self.pipe.vocoder.config.model_in_dim
        num_channels_latents = self.pipe.unet.config.in_channels

        latents = torch.randn(
            1,
            num_channels_latents,
            height // vae_scale_factor,
            model_in_dim // vae_scale_factor,
            dtype=dtype,
        )

        return {
            "sample": latents,
            "timestep": torch.tensor(1, dtype=torch.long),
            "encoder_hidden_states": None,
            "class_labels": prompt_embeds,
            "return_dict": False,
        }
