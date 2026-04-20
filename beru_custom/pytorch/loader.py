# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
beru_custom (jagat334433/beru_custom) model loader implementation.

beru_custom is a custom Stable Diffusion XL checkpoint distributed as
single-file safetensors in the jagat334433/beru_custom HuggingFace repo.

Available variants:
- BERU_CUSTOM_2: beru_custom_2.safetensors SDXL checkpoint
- BERU_CUSTOM_MERGE: beru_custom_merge.safetensors SDXL merged checkpoint
"""

from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline
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

REPO_ID = "jagat334433/beru_custom"


class ModelVariant(StrEnum):
    """Available beru_custom model variants."""

    BERU_CUSTOM_2 = "beru_custom_2"
    BERU_CUSTOM_MERGE = "beru_custom_merge"


_VARIANT_CHECKPOINTS = {
    ModelVariant.BERU_CUSTOM_2: "beru_custom_2.safetensors",
    ModelVariant.BERU_CUSTOM_MERGE: "beru_custom_merge.safetensors",
}


class ModelLoader(ForgeModel):
    """beru_custom SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.BERU_CUSTOM_2: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.BERU_CUSTOM_MERGE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.BERU_CUSTOM_2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="beru_custom",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the beru_custom SDXL pipeline from single-file checkpoint.

        Returns:
            StableDiffusionXLPipeline: The loaded pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        checkpoint_file = _VARIANT_CHECKPOINTS[self._variant]
        model_path = hf_hub_download(repo_id=REPO_ID, filename=checkpoint_file)
        self.pipeline = StableDiffusionXLPipeline.from_single_file(
            model_path,
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
