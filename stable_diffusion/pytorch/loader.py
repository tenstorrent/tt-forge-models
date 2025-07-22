# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion model loader implementation
https://huggingface.co/runwayml/stable-diffusion-v1-5
"""

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel
from PIL import Image
from diffusers import StableDiffusionPipeline
from ...tools.utils import get_file


class ModelLoader(ForgeModel):
    """Stable Diffusion model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "CompVis/stable-diffusion-v1-4"
        

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="stable_diffusion",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Stable Diffusion pipeline from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            StableDiffusionPipeline: The pre-trained Stable Diffusion pipeline object.
        """
        dtype = dtype_override or torch.bfloat16
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name, torch_dtype=dtype
        )
        return pipe


    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Stable Diffusion model.

        Args:
            dtype_override: This parameter is ignored for this model.
            batch_size: Optional batch size for the prompts.

        Returns:
            list: A list of sample text prompts.
        """

        prompt = [
            "a photo of an astronaut riding a horse on mars",
        ]
        return prompt