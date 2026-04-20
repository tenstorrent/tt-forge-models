# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AudioLDM medium model loader implementation for text-to-audio generation.
"""
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AudioLDMPipeline

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


class ModelVariant(StrEnum):
    """Available AudioLDM model variants."""

    M_FULL = "m-full"


class ModelLoader(ForgeModel):
    """AudioLDM medium model loader implementation for text-to-audio generation tasks."""

    _VARIANTS = {
        ModelVariant.M_FULL: ModelConfig(
            pretrained_model_name="cvssp/audioldm-m-full",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.M_FULL

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

        self.pipe.unet.eval()
        return self.pipe.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the AudioLDM UNet model."""
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32
        prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
        audio_length_in_s = 5.0

        # Encode text prompt using CLAP text encoder (projected + L2-normalized).
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

        # Convert audio length in seconds to spectrogram height matching the pipeline math.
        vocoder_upsample_factor = (
            np.prod(self.pipe.vocoder.config.upsample_rates)
            / self.pipe.vocoder.config.sampling_rate
        )
        height = int(audio_length_in_s / vocoder_upsample_factor)
        if height % self.pipe.vae_scale_factor != 0:
            height = (
                int(np.ceil(height / self.pipe.vae_scale_factor))
                * self.pipe.vae_scale_factor
            )

        shape = (
            1,
            self.pipe.unet.config.in_channels,
            height // self.pipe.vae_scale_factor,
            int(self.pipe.vocoder.config.model_in_dim) // self.pipe.vae_scale_factor,
        )
        latents = torch.randn(shape, dtype=dtype)
        timestep = torch.tensor(1, dtype=torch.int64)

        return {
            "sample": latents,
            "timestep": timestep,
            "encoder_hidden_states": None,
            "class_labels": prompt_embeds,
            "return_dict": False,
        }
