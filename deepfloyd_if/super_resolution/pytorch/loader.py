# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
DeepFloyd IF-II-L-v1.0 model loader implementation for super-resolution.

IF-II-L-v1.0 is the Stage II 64x64 -> 256x256 super-resolution diffusion
model in the DeepFloyd IF pipeline. It takes a low-resolution image and text
embeddings as conditioning to produce a higher-resolution image.
"""

import os
from typing import Optional

import torch

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available DeepFloyd IF super-resolution model variants."""

    IF_II_L_V1_0 = "IF-II-L-v1.0"


class ModelLoader(ForgeModel):
    """DeepFloyd IF-II-L-v1.0 model loader implementation for super-resolution."""

    _VARIANTS = {
        ModelVariant.IF_II_L_V1_0: ModelConfig(
            pretrained_model_name="DeepFloyd/IF-II-L-v1.0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.IF_II_L_V1_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DeepFloyd IF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self):
        from diffusers import IFSuperResolutionPipeline

        token = os.environ.get("HF_TOKEN")

        self._pipeline = IFSuperResolutionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=torch.float32,
            token=token,
        )
        self._pipeline.to("cpu")
        return self._pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        if self._pipeline is None:
            self._load_pipeline()

        unet = self._pipeline.unet
        unet.eval()

        if dtype_override is not None:
            unet = unet.to(dtype_override)

        return unet

    def load_inputs(self, dtype_override=None):
        if self._pipeline is None:
            self._load_pipeline()

        dtype = dtype_override or torch.float32

        pipe = self._pipeline

        # Encode a text prompt using the pipeline's text encoder
        prompt = "a photo of a kangaroo wearing an orange hoodie and blue sunglasses"
        prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

        unet = pipe.unet
        # Stage II UNet expects 6 input channels: 3 noisy sample + 3 upscaled low-res image
        in_channels = unet.config.in_channels
        sample_size = unet.config.sample_size

        sample = torch.randn(
            (1, in_channels, sample_size, sample_size),
            dtype=dtype,
        )

        timestep = torch.tensor([1], dtype=torch.long)

        # Stage II conditions on a noise level class label
        noise_level = torch.tensor([250], dtype=torch.long)

        return {
            "sample": sample.to(dtype),
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds.to(dtype),
            "class_labels": noise_level,
        }
