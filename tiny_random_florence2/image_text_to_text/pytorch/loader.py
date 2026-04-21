# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tiny Random Florence-2 model loader implementation for image-text-to-text tasks.
"""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

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
from ....tools.utils import get_file


class ModelVariant(StrEnum):
    """Available Tiny Random Florence-2 model variants."""

    TINY_RANDOM_FLORENCE2 = "tiny-random-Florence2ForConditionalGeneration"


class ModelLoader(ForgeModel):
    """Tiny Random Florence-2 model loader for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM_FLORENCE2: ModelConfig(
            pretrained_model_name="Xenova/tiny-random-Florence2ForConditionalGeneration",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM_FLORENCE2

    sample_text = "<CAPTION>"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="TinyRandomFlorence2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Tiny Random Florence-2 model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Florence2ForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            attn_implementation="eager",
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for Tiny Random Florence-2."""
        if self.processor is None:
            self._load_processor()

        image_path = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_path)).convert("RGB")

        inputs = self.processor(
            text=self.sample_text, images=image, return_tensors="pt"
        )

        # Florence-2 is a seq2seq model that requires decoder_input_ids
        decoder_start_token_id = self.processor.tokenizer.bos_token_id or 2
        inputs["decoder_input_ids"] = torch.full(
            (1, 1), decoder_start_token_id, dtype=torch.long
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
