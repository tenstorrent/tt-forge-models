# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ShieldGemma 2 model loader implementation for image safety classification.
"""

from typing import Optional

from PIL import Image
from transformers import AutoProcessor, ShieldGemma2ForImageClassification

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
from ...tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available ShieldGemma 2 model variants."""

    SHIELDGEMMA_2_4B_IT = "google/shieldgemma-2-4b-it"


class ModelLoader(ForgeModel):
    """ShieldGemma 2 model loader for image safety classification."""

    _VARIANTS = {
        ModelVariant.SHIELDGEMMA_2_4B_IT: ModelConfig(
            pretrained_model_name=str(ModelVariant.SHIELDGEMMA_2_4B_IT),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SHIELDGEMMA_2_4B_IT

    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ShieldGemma2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ShieldGemma 2 model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = ShieldGemma2ForImageClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for ShieldGemma 2."""
        if self.processor is None:
            self._load_processor()

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        inputs = self.processor(images=[image], return_tensors="pt")

        if dtype_override is not None:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs
