# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo2 model loader implementation for image-text-to-text (image-text generation).

Molmo2-8B is a multimodal vision-language model from AllenAI built on a
Qwen3-8B text decoder with a SigLIP-style ViT vision tower and an MLP adapter.
The checkpoint ships custom modeling/processing code (``trust_remote_code``),
authored against transformers 4.57.x — see the family ``requirements.txt`` for
the required pin.
"""
import io
from typing import Optional

import requests
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

# Pinned revision so the (custom-code) modeling/processing files are reproducible.
_MOLMO2_8B_REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"


class ModelVariant(StrEnum):
    """Available Molmo2 model variants for image-text generation."""

    MOLMO2_8B = "8b"


class ModelLoader(ForgeModel):
    """Molmo2 model loader implementation for image-text-to-text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.MOLMO2_8B: LLMModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

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
        """Load the processor for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the processor's default dtype.

        Returns:
            The loaded processor instance.
        """
        # The slow image processor is the one saved with the checkpoint; force it
        # to avoid the fast-processor numerical drift warning / mismatch.
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            revision=_MOLMO2_8B_REVISION,
            trust_remote_code=True,
            use_fast=False,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Molmo2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model uses its checkpoint dtype.

        Returns:
            torch.nn.Module: The Molmo2 model instance for image-text generation.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name,
            revision=_MOLMO2_8B_REVISION,
            trust_remote_code=True,
            **model_kwargs,
        )
        self.config = model.config
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Molmo2 model.

        Args:
            dtype_override: Optional torch.dtype to override the float input tensors' dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Fetch the sample image once and hand the processor a concrete PIL image
        # (rather than relying on the chat template to fetch a URL at run time).
        response = requests.get(self.sample_image_url, timeout=30)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
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

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Cast floating-point inputs (e.g. pixel_values) to the override dtype.
        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and torch.is_floating_point(inputs[key]):
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs

    def load_config(self):
        """Load and return the configuration for the Molmo2 model variant.

        Returns:
            The configuration object for the Molmo2 model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            revision=_MOLMO2_8B_REVISION,
            trust_remote_code=True,
        )
        return self.config
