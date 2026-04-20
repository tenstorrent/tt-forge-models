# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PaliGemma2 model loader implementation for image-text-to-text generation.
"""

import logging
import os
from typing import Optional

from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

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
from ....tools.utils import cast_input_to_type, get_file

logger = logging.getLogger(__name__)

_GATED_FALLBACKS = {
    "google/paligemma2-3b-pt-448": "kaischue/paligemma2-3b-pt-448-ACLFigQA-de",
}


def _get_hf_token():
    return os.environ.get("HF_TOKEN")


class ModelVariant(StrEnum):
    """Available PaliGemma2 model variants."""

    PALIGEMMA2_3B_PT_448 = "google/paligemma2-3b-pt-448"


class ModelLoader(ForgeModel):
    """PaliGemma2 model loader for image-text-to-text generation."""

    _VARIANTS = {
        ModelVariant.PALIGEMMA2_3B_PT_448: ModelConfig(
            pretrained_model_name=str(ModelVariant.PALIGEMMA2_3B_PT_448),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PALIGEMMA2_3B_PT_448

    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
    sample_text = ""

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self._resolved_model_name = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PaliGemma2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_resolved_model_name(self):
        if self._resolved_model_name is not None:
            return self._resolved_model_name
        name = self._variant_config.pretrained_model_name
        token = _get_hf_token()
        try:
            AutoProcessor.from_pretrained(name, token=token)
            self._resolved_model_name = name
        except OSError:
            fallback = _GATED_FALLBACKS.get(name)
            if fallback is None:
                raise
            logger.warning(
                "Gated repo %s inaccessible, falling back to %s",
                name,
                fallback,
            )
            self._resolved_model_name = fallback
        return self._resolved_model_name

    def _load_processor(self):
        model_name = self._get_resolved_model_name()
        token = _get_hf_token()
        self.processor = AutoProcessor.from_pretrained(model_name, token=token)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the PaliGemma2 model instance."""
        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["token"] = _get_hf_token()
        model_kwargs |= kwargs

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            self._get_resolved_model_name(), **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for PaliGemma2."""
        if self.processor is None:
            self._load_processor()

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        inputs = self.processor(
            text=self.sample_text,
            images=[image],
            return_tensors="pt",
        )

        if dtype_override is not None:
            inputs["pixel_values"] = cast_input_to_type(
                inputs["pixel_values"], dtype_override
            )

        return inputs
