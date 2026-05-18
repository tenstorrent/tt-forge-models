# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 1.4 model loader implementation.

Returns the ``UNet2DConditionModel`` (an ``nn.Module``) directly from
``load_model`` — the format the tt-xla model tester expects. The tokenizer,
text encoder and scheduler needed to build a sample input batch are kept on
the loader instance and used by ``load_inputs``.

The previous implementation of this loader returned the full
``StableDiffusionPipeline`` and referenced ``torch`` without importing it,
which made it unusable end-to-end; this rewrite mirrors the convention used
by ``stable_diffusion_unet`` and the new ``stable_diffusion_1_5`` loader.
"""

from typing import Optional

import torch
from diffusers import LMSDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

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
    """Available Stable Diffusion 1.4 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Stable Diffusion 1.4 model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="CompVis/stable-diffusion-v1-4",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with the requested variant.

        Args:
            variant: Optional ``ModelVariant``; falls back to ``DEFAULT_VARIANT``.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Stable Diffusion 1.4",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,  # FIXME: Update to text to image
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SD1.4 UNet.

        Args:
            dtype_override: Optional ``torch.dtype`` for the UNet weights;
                defaults to ``torch.bfloat16`` to match TT execution.

        Returns:
            torch.nn.Module: The ``UNet2DConditionModel`` instance for SD1.4.
        """
        dtype = dtype_override or torch.bfloat16

        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14", **kwargs
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14", **kwargs
        )
        unet = UNet2DConditionModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="unet",
            torch_dtype=dtype,
            **kwargs,
        )
        self.scheduler = LMSDiscreteScheduler.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="scheduler",
            **kwargs,
        )

        self.in_channels = unet.in_channels
        return unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Return a single-step UNet sample input batch for SD1.4.

        Args:
            dtype_override: Optional ``torch.dtype``; defaults to ``torch.bfloat16``.
            batch_size: Repetition factor for the prompt.

        Returns:
            dict: ``{"sample": …, "timestep": 0, "encoder_hidden_states": …}``.
        """
        dtype = dtype_override or torch.bfloat16

        prompt = ["A fantasy landscape with mountains and rivers"] * batch_size
        text_input = self.tokenizer(prompt, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids)[0]

        height, width = 512, 512
        latents = torch.randn((batch_size, self.in_channels, height // 8, width // 8))

        num_inference_steps = 1
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma
        latent_model_input = self.scheduler.scale_model_input(latents, 0)

        return {
            "sample": latent_model_input.to(dtype),
            "timestep": 0,
            "encoder_hidden_states": text_embeddings.to(dtype),
        }
