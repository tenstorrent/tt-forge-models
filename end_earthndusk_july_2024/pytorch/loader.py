# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
END EarthnDusk July 2024 (GoofyDinerMickeyDs/END-EarthnDuskJuly_2024) model loader.

A collection of Stable Diffusion v1.5 merged/fine-tuned single-file checkpoints
for text-to-image generation, distributed as .safetensors files.

Available variants:
- RANDOM_SD15_MERGE_001: RandomSD15Merge-001.fp16 text-to-image generation
"""

from typing import Optional

import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download

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

REPO_ID = "GoofyDinerMickeyDs/END-EarthnDuskJuly_2024"
CHECKPOINT_FILE = "RandomSD15Merge-001.fp16.safetensors"


class ModelVariant(StrEnum):
    """Available END EarthnDusk July 2024 model variants."""

    RANDOM_SD15_MERGE_001 = "RandomSD15Merge-001.fp16"


class ModelLoader(ForgeModel):
    """END EarthnDusk July 2024 model loader implementation."""

    _VARIANTS = {
        ModelVariant.RANDOM_SD15_MERGE_001: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.RANDOM_SD15_MERGE_001

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="END EarthnDusk July 2024",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the SD1.5 pipeline from a single-file checkpoint and return the UNet.

        Returns:
            UNet2DConditionModel: The UNet component of the pipeline.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        ckpt_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=CHECKPOINT_FILE,
        )
        self.pipeline = StableDiffusionPipeline.from_single_file(
            ckpt_path,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample UNet inputs with random tensors.

        Returns:
            dict: UNet inputs with sample latents, timestep, and encoder hidden states.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        unet = self.pipeline.unet
        in_channels = unet.config.in_channels
        cross_attention_dim = unet.config.cross_attention_dim

        latents = torch.randn(batch_size, in_channels, 64, 64, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, 77, cross_attention_dim, dtype=dtype
        )

        return {
            "sample": latents,
            "timestep": 0,
            "encoder_hidden_states": encoder_hidden_states,
        }
