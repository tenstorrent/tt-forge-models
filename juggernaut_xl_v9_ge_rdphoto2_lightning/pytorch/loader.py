# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Juggernaut XL V9 GE RDPhoto2 Lightning 4S
(AiWise/Juggernaut-XL-V9-GE-RDPhoto2-Lightning_4S) model loader implementation.

A photorealistic text-to-image model based on Stable Diffusion XL, merged with
RDPhoto2 and distilled with SDXL Lightning for 4-step inference. Distributed
as a single safetensors checkpoint derived from
stabilityai/stable-diffusion-xl-base-1.0.

Available variants:
- JUGGERNAUT_XL_V9_GE_RDPHOTO2_LIGHTNING_4S: single-file SDXL checkpoint.
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

REPO_ID = "AiWise/Juggernaut-XL-V9-GE-RDPhoto2-Lightning_4S"
CHECKPOINT_FILE = "juggernautXL_v9Rdphoto2Lightning.safetensors"


class ModelVariant(StrEnum):
    """Available Juggernaut XL V9 GE RDPhoto2 Lightning 4S model variants."""

    JUGGERNAUT_XL_V9_GE_RDPHOTO2_LIGHTNING_4S = (
        "Juggernaut_XL_V9_GE_RDPhoto2_Lightning_4S"
    )


class ModelLoader(ForgeModel):
    """Juggernaut XL V9 GE RDPhoto2 Lightning 4S model loader implementation."""

    _VARIANTS = {
        ModelVariant.JUGGERNAUT_XL_V9_GE_RDPHOTO2_LIGHTNING_4S: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.JUGGERNAUT_XL_V9_GE_RDPHOTO2_LIGHTNING_4S

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Juggernaut_XL_V9_GE_RDPhoto2_Lightning_4S",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Juggernaut XL V9 Lightning pipeline from single-file checkpoint.

        Returns:
            StableDiffusionXLPipeline: The loaded pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        model_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename=CHECKPOINT_FILE,
        )
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
