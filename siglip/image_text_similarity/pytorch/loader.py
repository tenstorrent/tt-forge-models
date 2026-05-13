# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SigLIP model loader implementation for image-text similarity.
"""

from typing import Optional

import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoModel, AutoProcessor

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available SigLIP model variants for image-text similarity."""

    BASE_PATCH16_224 = "Base_Patch16_224"
    BASE_PATCH16_256 = "Base_Patch16_256"
    BASE_PATCH16_384 = "Base_Patch16_384"
    BASE_PATCH16_512 = "Base_Patch16_512"
    BASE_PATCH16_256_MULTILINGUAL = "Base_Patch16_256_Multilingual"
    LARGE_PATCH16_256 = "Large_Patch16_256"
    LARGE_PATCH16_384 = "Large_Patch16_384"
    SO400M_PATCH14_224 = "So400m_Patch14_224"
    SO400M_PATCH14_384 = "So400m_Patch14_384"
    SO400M_PATCH16_256_I18N = "So400m_Patch16_256_I18n"


class ModelLoader(ForgeModel):
    """SigLIP model loader implementation for image-text similarity tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE_PATCH16_224: ModelConfig(
            pretrained_model_name="google/siglip-base-patch16-224",
        ),
        ModelVariant.BASE_PATCH16_256: ModelConfig(
            pretrained_model_name="google/siglip-base-patch16-256",
        ),
        ModelVariant.BASE_PATCH16_384: ModelConfig(
            pretrained_model_name="google/siglip-base-patch16-384",
        ),
        ModelVariant.BASE_PATCH16_512: ModelConfig(
            pretrained_model_name="google/siglip-base-patch16-512",
        ),
        ModelVariant.BASE_PATCH16_256_MULTILINGUAL: ModelConfig(
            pretrained_model_name="google/siglip-base-patch16-256-multilingual",
        ),
        ModelVariant.LARGE_PATCH16_256: ModelConfig(
            pretrained_model_name="google/siglip-large-patch16-256",
        ),
        ModelVariant.LARGE_PATCH16_384: ModelConfig(
            pretrained_model_name="google/siglip-large-patch16-384",
        ),
        ModelVariant.SO400M_PATCH14_224: ModelConfig(
            pretrained_model_name="google/siglip-so400m-patch14-224",
        ),
        ModelVariant.SO400M_PATCH14_384: ModelConfig(
            pretrained_model_name="google/siglip-so400m-patch14-384",
        ),
        ModelVariant.SO400M_PATCH16_256_I18N: ModelConfig(
            pretrained_model_name="google/siglip-so400m-patch16-256-i18n",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE_PATCH16_224

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.processor = None
        self.text_prompts = None

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
            model="SigLIP",
            variant=variant,
            group=(
                ModelGroup.RED
                if variant == ModelVariant.BASE_PATCH16_224
                else ModelGroup.GENERALITY
            ),
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        # Load the processor
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SigLIP model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The SigLIP model instance for image-text similarity.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}

        # Load the model with dtype override if specified
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SigLIP model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Load image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        # Define text prompts for image-text similarity
        self.text_prompts = ["a photo of 2 cats", "a photo of 2 dogs"]

        # Process both text and images
        inputs = self.processor(
            text=self.text_prompts,
            images=image,
            return_tensors="pt",
            padding="max_length",
        )

        # Replicate tensors for batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Convert the input dtype to dtype_override if specified
        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs

    def unpack_forward_output(self, forward_output):
        """Unpack forward pass output to a single differentiable tensor.

        Forward output structure (with return_dict=False):
            tuple(6):
              [0] Tensor (batch_image, batch_text) — logits_per_image.
              [1] Tensor (batch_text, batch_image) — logits_per_text.
              [2] Tensor (batch_text, hidden_dim) — text_embeds.
              [3] Tensor (batch_image, hidden_dim) — image_embeds.
              [4] tuple — text_outputs (last_hidden_state, pooler_output).
              [5] tuple — vision_outputs (last_hidden_state, pooler_output).

        What is selected and why:
            SigLIP's contrastive sigmoid loss is computed from the
            image-text logit pairs. logits_per_image and logits_per_text are
            transposes of the same dot-product matrix, so either is sufficient
            as a loss gradient source. We return logits_per_text to mirror the
            existing CLIPOutput → logits_per_text registry entry. The text /
            vision encoder outputs and projected embeddings are intermediates
            and excluded.
        """
        return forward_output[1]
