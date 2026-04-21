# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KORMo-VL model loader implementation for image-text-to-text document
understanding and bilingual (Korean/English) vision-language tasks.
"""
from typing import Optional

from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

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
from ....tools.utils import get_file


class ModelVariant(StrEnum):
    """Available KORMo-VL model variants."""

    KORMO_VL = "kormo_vl"


class ModelLoader(ForgeModel):
    """KORMo-VL model loader for image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.KORMO_VL: ModelConfig(
            pretrained_model_name="KORMo-VL/KORMo-VL",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KORMO_VL

    sample_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
    sample_text = "이 이미지에 대해 설명해주세요."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="kormo_vl",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.sample_text},
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

        if dtype_override is not None:
            for key in inputs:
                if inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype_override)

        if batch_size > 1:
            for key in inputs:
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return dict(inputs)

    def decode_output(self, outputs, input_length=None):
        if self.processor is None:
            self._load_processor()

        if hasattr(outputs, "logits"):
            predicted_ids = outputs.logits.argmax(-1)
        else:
            predicted_ids = outputs

        if input_length is not None:
            predicted_ids = predicted_ids[:, input_length:]

        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
