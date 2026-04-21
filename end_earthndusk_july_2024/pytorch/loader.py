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
        """Load and return the SD1.5 pipeline from a single-file checkpoint.

        Returns:
            StableDiffusionPipeline: The pipeline instance.
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
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
