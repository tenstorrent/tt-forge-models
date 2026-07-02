# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Molmo2 model loader implementation for image-text-to-text (vision-language).

allenai/Molmo2-8B is a vision-language model: a SigLIP-style ViT vision tower + a
pooling adapter projecting into an Olmo/Qwen3-8B text decoder. It is distributed as
Hub ``custom_code``; ``src/molmo2_compat.py`` documents the small shims needed to run
it under the venv's pinned ``transformers==5.5.1`` (see that file for the rationale).
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
from .src.molmo2_compat import (
    MOLMO2_REVISION,
    apply_model_shims,
    apply_processor_shims,
    propagate_top_level_config_attrs,
)


class ModelVariant(StrEnum):
    """Available Molmo2 model variants for image-text-to-text."""

    MOLMO2_8B = "8b"


class ModelLoader(ForgeModel):
    """Molmo2 model loader implementation for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.MOLMO2_8B: LLMModelConfig(
            pretrained_model_name="allenai/Molmo2-8B",
            max_length=1024,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOLMO2_8B

    # Fixed sample image + prompt so CPU and device see identical inputs.
    sample_image_url = (
        "https://huggingface.co/datasets/huggingface/documentation-images/"
        "resolve/main/pipeline-cat-chonk.jpeg"
    )
    sample_text = "Describe this image."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

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
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load (and cache) the Molmo2 processor with 5.x compat shims applied."""
        if self.processor is None:
            apply_processor_shims(revision=MOLMO2_REVISION)
            self.processor = AutoProcessor.from_pretrained(
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
                revision=MOLMO2_REVISION,
            )
        return self.processor

    def load_model(self, dtype_override=None, **kwargs):
        """Load and return the Molmo2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model uses its checkpoint dtype
                            (float32).

        Returns:
            torch.nn.Module: The Molmo2 model instance for image-text-to-text.
        """
        apply_model_shims()
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Build the config and propagate the top-level attrs the forward reads but
        # which now live only on text_config under transformers 5.x.
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True, revision=MOLMO2_REVISION
        )
        propagate_top_level_config_attrs(config)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name,
            config=config,
            trust_remote_code=True,
            revision=MOLMO2_REVISION,
            **model_kwargs,
            **kwargs,
        ).eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample image+text inputs for the Molmo2 model.

        Args:
            dtype_override: Optional torch.dtype to cast floating-point inputs
                            (pixel values) to, matching the model dtype.
            batch_size: Batch size for the inputs (only 1 is supported for the
                        multi-crop vision path).

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        processor = self._load_processor()

        image = Image.open(
            io.BytesIO(requests.get(self.sample_image_url, timeout=60).content)
        ).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.sample_text},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # token_type_ids is emitted by the processor but not accepted by the forward.
        inputs.pop("token_type_ids", None)

        # Cast floating-point inputs (pixel values) to the model dtype.
        if dtype_override is not None:
            for key, value in inputs.items():
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    inputs[key] = value.to(dtype_override)

        return dict(inputs)
