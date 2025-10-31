# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 3.5 model loader implementation
https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
"""

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
import torch
from diffusers import StableDiffusion3Pipeline
from typing import Optional


class ModelVariant(StrEnum):
    """Available Stable Diffusion 3.5 text encoder variants."""

    MEDIUM_1 = "medium-1"
    MEDIUM_2 = "medium-2"
    MEDIUM_3 = "medium-3"
    LARGE_1 = "large-1"
    LARGE_2 = "large-2"
    LARGE_3 = "large-3"


class ModelLoader(ForgeModel):
    """Stable Diffusion 3.5 text encoder model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.MEDIUM_1: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3.5-medium",
        ),
        ModelVariant.MEDIUM_2: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3.5-medium",
        ),
        ModelVariant.MEDIUM_3: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3.5-medium",
        ),
        ModelVariant.LARGE_1: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3.5-large",
        ),
        ModelVariant.LARGE_2: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3.5-large",
        ),
        ModelVariant.LARGE_3: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3.5-large",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MEDIUM_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="stable_diffusion_3_5",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Stable Diffusion 3.5 text encoder for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            The appropriate text encoder (text_encoder, text_encoder_2, or text_encoder_3).
        """
        model_path = self._variant_config.pretrained_model_name
        pipe_kwargs = {}
        if dtype_override is not None:
            pipe_kwargs["torch_dtype"] = dtype_override
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **pipe_kwargs,
        )

        # Extract encoder type from variant
        if (
            self._variant == ModelVariant.MEDIUM_1
            or self._variant == ModelVariant.LARGE_1
        ):
            return self.pipe.text_encoder
        elif (
            self._variant == ModelVariant.MEDIUM_2
            or self._variant == ModelVariant.LARGE_2
        ):
            return self.pipe.text_encoder_2
        else:  # default to text_encoder_3
            return self.pipe.text_encoder_3

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the Stable Diffusion 3.5 text encoder model.

        Args:
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing input_ids and attention_mask.
        """
        prompt = "A futuristic cityscape at sunset"

        # Get the corresponding tokenizer based on variant
        if (
            self._variant == ModelVariant.MEDIUM_1
            or self._variant == ModelVariant.LARGE_1
        ):
            tokenizer = self.pipe.tokenizer
        elif (
            self._variant == ModelVariant.MEDIUM_2
            or self._variant == ModelVariant.LARGE_2
        ):
            tokenizer = self.pipe.tokenizer_2
        else:  # default to tokenizer_3
            tokenizer = self.pipe.tokenizer_3

        # Tokenize the prompt
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Support batch size by repeating inputs
        arguments = {
            "input_ids": text_inputs.input_ids.repeat(batch_size, 1),
            "attention_mask": text_inputs.attention_mask.repeat(batch_size, 1),
        }

        return arguments
