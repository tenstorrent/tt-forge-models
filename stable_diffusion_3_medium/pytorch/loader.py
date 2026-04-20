# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 3 Medium model loader implementation.

Loads single-file safetensors checkpoints from the original
stabilityai/stable-diffusion-3-medium repository.
"""

from typing import Optional

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
from .src.model_utils import load_pipe, stable_diffusion_preprocessing_v3


class ModelVariant(StrEnum):
    """Available Stable Diffusion 3 Medium single-file variants."""

    INCL_CLIPS_T5XXL_FP16 = "incl_clips_t5xxlfp16"


class ModelLoader(ForgeModel):
    """Stable Diffusion 3 Medium model loader implementation."""

    _VARIANTS = {
        ModelVariant.INCL_CLIPS_T5XXL_FP16: ModelConfig(
            pretrained_model_name="sd3_medium_incl_clips_t5xxlfp16.safetensors",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INCL_CLIPS_T5XXL_FP16

    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Stable Diffusion 3 Medium",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Stable Diffusion 3 Medium transformer for this variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Stable Diffusion 3 transformer instance.
        """
        filename = self._variant_config.pretrained_model_name

        self.pipeline = load_pipe(filename)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Stable Diffusion 3 Medium model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors for the transformer:
                - latent_model_input (torch.Tensor)
                - timestep (torch.Tensor)
                - prompt_embeds (torch.Tensor)
                - pooled_prompt_embeds (torch.Tensor)
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            latent_model_input,
            timestep,
            prompt_embeds,
            pooled_prompt_embeds,
        ) = stable_diffusion_preprocessing_v3(self.pipeline, self.prompt)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype_override)

        return [latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds]
