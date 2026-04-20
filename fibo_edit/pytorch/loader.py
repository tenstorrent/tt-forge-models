# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Fibo-Edit model loader implementation for image-to-image editing.

Fibo-Edit is an 8B-parameter DiT-based, flow-matching image editing model
from Bria AI. It consumes a source image plus a structured JSON prompt and
produces an edited image via the BriaFiboEditPipeline from diffusers.
"""

from typing import Optional

import torch
from diffusers import DiffusionPipeline
from PIL import Image

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
    """Available Fibo-Edit model variants."""

    FIBO_EDIT = "fibo_edit"


class ModelLoader(ForgeModel):
    """Fibo-Edit model loader implementation for image-to-image editing tasks."""

    _VARIANTS = {
        ModelVariant.FIBO_EDIT: ModelConfig(
            pretrained_model_name="briaai/Fibo-Edit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FIBO_EDIT

    prompt = '{"instruction": "make it look vintage"}'

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="fibo_edit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Fibo-Edit pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            DiffusionPipeline: The Fibo-Edit pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Fibo-Edit model.

        Returns:
            dict: Dictionary containing the source image and structured JSON prompt.
        """
        image = Image.new("RGB", (512, 512), color=(180, 140, 100))
        return {
            "image": image,
            "prompt": [self.prompt] * batch_size,
        }
