# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OWLv2 model loader implementation for zero-shot object detection.
"""

import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from datasets import load_dataset
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available OWLv2 model variants for zero-shot object detection."""

    BASE_PATCH16_FINETUNED = "Base_Patch16_Finetuned"


class ModelLoader(ForgeModel):
    """OWLv2 model loader implementation for zero-shot object detection tasks."""

    _VARIANTS = {
        ModelVariant.BASE_PATCH16_FINETUNED: ModelConfig(
            pretrained_model_name="google/owlv2-base-patch16-finetuned",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_PATCH16_FINETUNED

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.image = None
        self.text_labels = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="OWLv2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_ZS_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant."""
        self.processor = Owlv2Processor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the OWLv2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The OWLv2 model instance for zero-shot object detection.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Owlv2ForObjectDetection.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the OWLv2 model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        self.image = dataset[0]["image"]

        self.text_labels = [["a photo of a cat", "a photo of a dog"]]

        inputs = self.processor(
            text=self.text_labels, images=self.image, return_tensors="pt"
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
