# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite Vision 3.3 2B GGUF model loader implementation for image-text-to-text tasks.
"""
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
from transformers.image_utils import load_image
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

# The GGUF file only contains the language backbone (no vision encoder weights).
# Transformers does not support the 'granite' GGUF architecture natively, and
# test-collection-time patches to load_gguf_checkpoint from other models break
# the GGUF loading path.  Load model and config from the non-quantised base
# repo so the full multimodal graph is available for compilation.
BASE_MODEL = "ibm-granite/granite-vision-3.3-2b"


class ModelVariant(StrEnum):
    """Available Granite Vision 3.3 2B GGUF model variants for image-text-to-text tasks."""

    GRANITE_VISION_3_3_2B_GGUF = "3.3_2B_GGUF"


class ModelLoader(ForgeModel):
    """Granite Vision 3.3 2B GGUF model loader implementation for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.GRANITE_VISION_3_3_2B_GGUF: LLMModelConfig(
            pretrained_model_name=BASE_MODEL,
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRANITE_VISION_3_3_2B_GGUF

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Granite Vision 3.3 2B GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(BASE_MODEL, **kwargs)

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What is in this image?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(BASE_MODEL)
        return self.config
