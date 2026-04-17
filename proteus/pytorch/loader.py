# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Proteus V0.3 model loader implementation

Proteus V0.3 is a text-to-image model based on Stable Diffusion XL,
fine-tuned by dataautogpt3 on high-quality captioned images with DPO.
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
    """Available Proteus model variants."""

    PROTEUS_V0_3 = "ProteusV0.3"


class ModelLoader(ForgeModel):
    """Proteus V0.3 model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.PROTEUS_V0_3: ModelConfig(
            pretrained_model_name="dataautogpt3/ProteusV0.3",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.PROTEUS_V0_3

    # Shared configuration parameters
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
            model="Proteus",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Proteus UNet for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The UNet component from the Proteus pipeline.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Load the pipeline
        self.pipeline = load_pipe(pretrained_model_name)

        # Apply dtype conversion if specified
        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Proteus UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Keyword arguments for the UNet forward method:
                - sample (torch.Tensor): Latent input for the UNet
                - timestep (torch.Tensor): Single timestep tensor
                - encoder_hidden_states (torch.Tensor): Encoded prompt embeddings
                - added_cond_kwargs (dict): Additional conditioning inputs
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

        timestep = timesteps[0]

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "added_cond_kwargs": added_cond_kwargs,
        }
