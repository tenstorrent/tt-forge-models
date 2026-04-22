# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LightOnOCR-2-1B GGUF model loader implementation for image-to-text OCR tasks.

The GGUF checkpoint only contains the text backbone (qwen3 architecture).
We load the vision config from the base (non-GGUF) model and combine
them into a full LightOnOcrConfig.
"""

import importlib.metadata

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoConfig,
    LightOnOcrConfig,
)
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


def _refresh_gguf_detection():
    """Refresh transformers' gguf package detection if the package was installed after import."""
    from transformers.utils import import_utils

    if "gguf" not in import_utils.PACKAGE_DISTRIBUTION_MAPPING:
        import_utils.PACKAGE_DISTRIBUTION_MAPPING = (
            importlib.metadata.packages_distributions()
        )
        import_utils.is_gguf_available.cache_clear()


class ModelVariant(StrEnum):
    """Available LightOnOCR-2-1B GGUF model variants for image-to-text tasks."""

    LIGHTON_OCR_2_1B_GGUF = "lighton_ocr_2_1b_gguf"


class ModelLoader(ForgeModel):
    """LightOnOCR-2-1B GGUF model loader implementation for image-to-text OCR tasks."""

    _VARIANTS = {
        ModelVariant.LIGHTON_OCR_2_1B_GGUF: LLMModelConfig(
            pretrained_model_name="noctrex/LightOnOCR-2-1B-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LIGHTON_OCR_2_1B_GGUF

    GGUF_FILE = "LightOnOCR-2-1B-Q4_K_M.gguf"

    _BASE_PROCESSOR_MODEL = "lightonai/LightOnOCR-2-1B"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="lighton_ocr_2_1b_gguf",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _build_full_config(self):
        """Build a full LightOnOcrConfig by wrapping the GGUF text config with a vision config."""
        _refresh_gguf_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name
        text_config = AutoConfig.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        base_config = AutoConfig.from_pretrained(self._BASE_PROCESSOR_MODEL)
        return LightOnOcrConfig(
            text_config=text_config.to_dict(),
            vision_config=base_config.vision_config.to_dict(),
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _refresh_gguf_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self._BASE_PROCESSOR_MODEL)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs["gguf_file"] = self.GGUF_FILE
        model_kwargs |= kwargs

        config = self._build_full_config()
        model_kwargs["config"] = config

        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self._BASE_PROCESSOR_MODEL)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
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

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = self._build_full_config()
        return self.config
