# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Segmind-Vega model loader implementation.

Segmind-Vega is a distilled text-to-image diffusion model from Segmind that
is ~70% smaller than Stable Diffusion XL while retaining its architecture
and ``StableDiffusionXLPipeline`` interface.
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
from .src.model_utils import load_pipe
from ...stable_diffusion_xl.pytorch.src.model_utils import (
    stable_diffusion_preprocessing_xl,
)


class ModelVariant(StrEnum):
    """Available Segmind-Vega model variants."""

    SEGMIND_VEGA = "Segmind-Vega"


class ModelLoader(ForgeModel):
    """Segmind-Vega model loader implementation."""

    _VARIANTS = {
        ModelVariant.SEGMIND_VEGA: ModelConfig(
            pretrained_model_name="segmind/Segmind-Vega",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEGMIND_VEGA

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
        """Implementation method for getting model info with validated variant."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Segmind-Vega",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Segmind-Vega pipeline for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype
                           (typically float32).

        Returns:
            DiffusionPipeline: The Segmind-Vega pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Segmind-Vega model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs'
                           default dtype.

        Returns:
            list: Input tensors that can be fed to the UNet:
                - latent_model_input (torch.Tensor): Latent input for the UNet
                - timesteps (torch.Tensor): Timestep tensor
                - prompt_embeds (torch.Tensor): Encoded prompt embeddings
                - added_cond_kwargs (dict): Additional conditioning inputs
                  (text embeddings and time IDs).
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(self.pipeline, self.prompt)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timesteps, prompt_embeds, added_cond_kwargs]
