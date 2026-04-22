# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BK-SDM-Tiny model loader implementation.

Block-removed Knowledge-distilled Stable Diffusion Model (BK-SDM) is an
architecturally compressed SDM for efficient text-to-image synthesis. BK-SDM-Tiny
removes residual and attention blocks from the U-Net of Stable Diffusion v1.4
and is loaded via the standard StableDiffusionPipeline.
"""

import torch
from typing import Optional

from diffusers import StableDiffusionPipeline

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
    """Available BK-SDM-Tiny model variants."""

    BK_SDM_TINY = "bk-sdm-tiny"


class ModelLoader(ForgeModel):
    """BK-SDM-Tiny model loader implementation."""

    _VARIANTS = {
        ModelVariant.BK_SDM_TINY: ModelConfig(
            pretrained_model_name="nota-ai/bk-sdm-tiny",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BK_SDM_TINY

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="BK-SDM-Tiny",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BK-SDM-Tiny UNet from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            UNet2DConditionModel: The pre-trained BK-SDM-Tiny UNet.
        """
        dtype = dtype_override or torch.bfloat16
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype, **kwargs
        )
        return self.pipe.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the BK-SDM-Tiny UNet.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing sample, timestep, and encoder_hidden_states.
        """
        dtype = dtype_override or torch.bfloat16

        prompt = ["a tropical bird sitting on a branch of a tree"] * batch_size
        text_inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        encoder_hidden_states = self.pipe.text_encoder(text_inputs.input_ids)[0].to(
            dtype=dtype
        )

        vae_scale_factor = 2 ** (len(self.pipe.vae.config.block_out_channels) - 1)
        latent_height = 512 // vae_scale_factor
        latent_width = 512 // vae_scale_factor
        num_channels = self.pipe.unet.config.in_channels

        sample = torch.randn(
            batch_size, num_channels, latent_height, latent_width, dtype=dtype
        )
        timestep = torch.tensor([1], dtype=dtype)

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
