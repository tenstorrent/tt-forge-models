# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SD-Turbo model loader implementation
"""

import torch
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


class ModelVariant(StrEnum):
    """Available SD-Turbo model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """SD-Turbo model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="stabilityai/sd-turbo",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="SD-Turbo",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override or torch.bfloat16
        model_name = self._variant_config.pretrained_model_name

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer", **kwargs
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder", **kwargs
        )
        unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet", torch_dtype=dtype, **kwargs
        )
        self.in_channels = unet.config.in_channels
        self.sample_size = unet.config.sample_size
        return unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        dtype = dtype_override or torch.bfloat16

        prompt = [
            "A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
        ] * batch_size
        text_input = self.tokenizer(prompt, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids)[0]

        sample = torch.randn(
            (batch_size, self.in_channels, self.sample_size, self.sample_size)
        )
        timestep = torch.tensor([1], dtype=torch.long)

        return {
            "sample": sample.to(dtype),
            "timestep": timestep,
            "encoder_hidden_states": text_embeddings.to(dtype),
        }
