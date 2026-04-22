# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MLbackup Lora_scrape_2023 LoRA Stable Diffusion model loader implementation
"""

from typing import Any, Optional

import torch
from diffusers import StableDiffusionPipeline

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
    """Available MLbackup Lora_scrape_2023 model variants."""

    DARK_FANTASY_V2 = "Dark_FantasyV2"


class ModelLoader(ForgeModel):
    """MLbackup Lora_scrape_2023 LoRA Stable Diffusion model loader implementation."""

    _VARIANTS = {
        ModelVariant.DARK_FANTASY_V2: ModelConfig(
            pretrained_model_name="MLbackup/Lora_scrape_2023",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DARK_FANTASY_V2

    _LORA_WEIGHT_NAMES = {
        ModelVariant.DARK_FANTASY_V2: "Dark_FantasyV2.safetensors",
    }

    _BASE_MODEL = "runwayml/stable-diffusion-v1-5"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[StableDiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="MLbackup Lora_scrape_2023",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self._BASE_MODEL, torch_dtype=dtype, **kwargs
        )
        self.pipeline.load_lora_weights(
            self._variant_config.pretrained_model_name,
            weight_name=self._LORA_WEIGHT_NAMES[self._variant],
        )
        self.pipeline.fuse_lora()
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, **kwargs):
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = self.pipeline.unet.dtype

        prompt = "a beautiful fantasy illustration, detailed artwork, masterpiece"

        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_hidden_states = self.pipeline.text_encoder(text_inputs.input_ids)[
                0
            ].to(dtype)

        in_channels = self.pipeline.unet.config.in_channels
        sample_size = self.pipeline.unet.config.sample_size
        latent_sample = torch.randn(
            1, in_channels, sample_size, sample_size, dtype=dtype
        )
        timestep = torch.tensor([1.0], dtype=dtype)

        if dtype_override:
            latent_sample = latent_sample.to(dtype_override)
            timestep = timestep.to(dtype_override)
            encoder_hidden_states = encoder_hidden_states.to(dtype_override)

        return [latent_sample, timestep, encoder_hidden_states]

    def unpack_forward_output(self, fwd_output: Any) -> torch.Tensor:
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        if hasattr(fwd_output, "sample"):
            return fwd_output.sample
        return fwd_output
