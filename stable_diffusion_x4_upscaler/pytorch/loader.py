# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion x4 Upscaler model loader implementation.

Loads the stabilityai/stable-diffusion-x4-upscaler pipeline and extracts
the UNet2DConditionModel for compilation. The UNet performs latent upscaling
conditioned on text embeddings, with 7 input channels (4 latent + 3 low-res).

Available variants:
- BASE: stabilityai/stable-diffusion-x4-upscaler
"""

from typing import Any, Optional

import torch
from diffusers import StableDiffusionUpscalePipeline

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
    """Available Stable Diffusion x4 Upscaler model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Stable Diffusion x4 Upscaler model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-x4-upscaler",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Stable Diffusion x4 Upscaler",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override or torch.bfloat16
        self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        in_channels = 7
        sample_size = 128
        cross_attention_dim = 1024
        seq_len = 77

        sample = torch.randn(
            (batch_size, in_channels, sample_size, sample_size),
            dtype=dtype,
        )
        timestep = torch.randint(0, 1000, (1,))
        encoder_hidden_states = torch.randn(
            (batch_size, seq_len, cross_attention_dim),
            dtype=dtype,
        )

        class_labels = torch.tensor([20] * batch_size, dtype=torch.long)

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "class_labels": class_labels,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if hasattr(output, "sample"):
            return output.sample
        elif isinstance(output, tuple):
            return output[0]
        return output
