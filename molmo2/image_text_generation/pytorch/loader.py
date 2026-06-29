# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo2 model loader implementation for image-text-to-text tasks.

Molmo2-8B (allenai) is a vision-language model: a SigLIP-style vision tower +
an MLP adapter/connector + a Qwen3-8B-based text decoder. It ships as
``custom_code`` and requires ``trust_remote_code=True`` and ``transformers==4.57.1``
(see requirements.txt next to this loader).
"""
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Molmo2 model variants for image-text-to-text tasks."""

    MOLMO2_8B = "molmo2_8b"


class ModelLoader(ForgeModel):
    """Molmo2 model loader implementation for image-text-to-text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.MOLMO2_8B: LLMModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

    # Pin the HF revision so the custom modeling/processing code is reproducible.
    REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"

    # Shared configuration parameters
    sample_image_url = (
        "https://huggingface.co/datasets/huggingface/documentation-images/"
        "resolve/main/pipeline-cat-chonk.jpeg"
    )
    sample_prompt = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.model = None

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
            model="molmo2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        """Load the Molmo2 processor for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the processor's default dtype.

        Returns:
            The loaded processor instance
        """
        kwargs = {"trust_remote_code": True, "revision": self.REVISION}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Molmo2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model uses its default dtype (float32).

        Returns:
            torch.nn.Module: The Molmo2 model instance for image-text-to-text tasks.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True, "revision": self.REVISION}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.config = model.config
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Molmo2 model.

        Args:
            dtype_override: Optional torch.dtype to override the float inputs' dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.sample_image_url},
                    {"type": "text", "text": self.sample_prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Cast floating-point inputs (pixel_values) to the requested dtype.
        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and torch.is_floating_point(inputs[key]):
                    inputs[key] = inputs[key].to(dtype_override)

        # Replicate across the batch dimension if requested.
        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return dict(inputs)

    def load_config(self):
        """Load and return the configuration for the Molmo2 model variant.

        Returns:
            The configuration object for the Molmo2 model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            revision=self.REVISION,
        )
        return self.config
