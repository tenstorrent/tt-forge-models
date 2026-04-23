# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 3.5 Large TensorRT model loader implementation.

The stabilityai/stable-diffusion-3.5-large-tensorrt repository hosts ONNX
exports of stabilityai/stable-diffusion-3.5-large intended for TensorRT
engine compilation. Since the repository does not ship PyTorch weights, this
loader exercises the reference SD 3.5 Large transformer architecture loaded
from the adamo1139/stable-diffusion-3.5-large-ungated mirror (the base
stabilityai repo is gated).
"""

from typing import Optional

import torch
from diffusers import StableDiffusion3Pipeline

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
from ...stable_diffusion.pytorch.src.model_utils import (
    stable_diffusion_preprocessing_v35,
)


class ModelVariant(StrEnum):
    """Available Stable Diffusion 3.5 Large TensorRT model variants."""

    LARGE_TENSORRT = "Large_TensorRT"


class ModelLoader(ForgeModel):
    """Stable Diffusion 3.5 Large TensorRT model loader implementation."""

    _VARIANTS = {
        ModelVariant.LARGE_TENSORRT: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3.5-large-tensorrt",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_TENSORRT

    BASE_PIPELINE = "adamo1139/stable-diffusion-3.5-large-ungated"

    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Stable Diffusion 3.5 Large TensorRT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the SD 3.5 Large transformer used by the TensorRT export.

        Returns:
            torch.nn.Module: The SD3 transformer instance.
        """
        compute_dtype = dtype_override or torch.bfloat16

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.BASE_PIPELINE,
            torch_dtype=compute_dtype,
        )
        self.pipeline.to("cpu")

        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None):
        """Load sample inputs for the SD 3.5 Large TensorRT transformer.

        Returns:
            list: [latent_model_input, prompt_embeds, pooled_prompt_embeds, timestep]
                  matching SD3Transformer2DModel.forward positional arg order:
                  (hidden_states, encoder_hidden_states, pooled_projections, timestep)
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            latent_model_input,
            timestep,
            prompt_embeds,
            pooled_prompt_embeds,
        ) = stable_diffusion_preprocessing_v35(self.pipeline, self.prompt)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype_override)

        return [latent_model_input, prompt_embeds, pooled_prompt_embeds, timestep]
