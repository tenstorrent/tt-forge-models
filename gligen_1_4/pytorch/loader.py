# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLIGEN 1.4 model loader implementation

GLIGEN extends Stable Diffusion with grounded text-to-image generation,
allowing placement of objects at specified bounding box locations.

Reference: https://huggingface.co/masterful/gligen-1-4-generation-text-box
"""

from typing import Optional

import torch
from diffusers import StableDiffusionGLIGENPipeline

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
from .src.model_utils import gligen_preprocessing


class ModelVariant(StrEnum):
    """Available GLIGEN model variants."""

    GENERATION_TEXT_BOX = "generation-text-box"


class ModelLoader(ForgeModel):
    """GLIGEN 1.4 model loader implementation."""

    _VARIANTS = {
        ModelVariant.GENERATION_TEXT_BOX: ModelConfig(
            pretrained_model_name="masterful/gligen-1-4-generation-text-box",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GENERATION_TEXT_BOX

    _prompt = "a waterfall and a modern high speed train in a beautiful forest with fall foliage"
    _gligen_phrases = [["a waterfall", "a modern high speed train"]]
    _gligen_boxes = [
        [[0.1387, 0.2051, 0.4277, 0.7090], [0.4980, 0.4355, 0.8516, 0.7266]]
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GLIGEN 1.4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the GLIGEN pipeline and return its UNet as a torch.nn.Module.

        Returns:
            torch.nn.Module: The UNet used for denoising.
        """
        dtype = dtype_override or torch.float32
        self.pipeline = StableDiffusionGLIGENPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        self.pipeline.enable_fuser(True)
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load tensor inputs for a single GLIGEN UNet forward pass.

        Returns:
            dict: keyword arguments for the UNet forward method.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        prompt = [self._prompt] * batch_size
        gligen_phrases = self._gligen_phrases * batch_size
        gligen_boxes = self._gligen_boxes * batch_size

        (
            latent_model_input,
            t,
            prompt_embeds,
            cross_attention_kwargs,
        ) = gligen_preprocessing(
            self.pipeline,
            prompt,
            gligen_phrases,
            gligen_boxes,
        )

        if dtype_override is not None:
            latent_model_input = latent_model_input.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return {
            "sample": latent_model_input,
            "timestep": t,
            "encoder_hidden_states": prompt_embeds,
            "cross_attention_kwargs": cross_attention_kwargs,
        }
