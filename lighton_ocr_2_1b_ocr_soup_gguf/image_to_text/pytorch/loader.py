# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LightOnOCR-2-1B ocr-soup GGUF model loader implementation for image-to-text OCR tasks.

The GGUF checkpoint only contains the text backbone (qwen3 architecture).
Load it as Qwen3ForCausalLM, then transplant its weights into the full VL
model loaded from the base repo so the vision encoder is included.
"""

import importlib.metadata

import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    LightOnOcrForConditionalGeneration,
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
    """Available LightOnOCR-2-1B ocr-soup GGUF model variants for image-to-text tasks."""

    LIGHTON_OCR_2_1B_OCR_SOUP_GGUF = "lighton_ocr_2_1b_ocr_soup_gguf"


class ModelLoader(ForgeModel):
    """LightOnOCR-2-1B ocr-soup GGUF model loader implementation for image-to-text OCR tasks."""

    _VARIANTS = {
        ModelVariant.LIGHTON_OCR_2_1B_OCR_SOUP_GGUF: LLMModelConfig(
            pretrained_model_name="noctrex/LightOnOCR-2-1B-ocr-soup-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LIGHTON_OCR_2_1B_OCR_SOUP_GGUF

    GGUF_FILE = "LightOnOCR-2-1B-ocr-soup-Q4_K_M.gguf"

    _BASE_PROCESSOR_MODEL = "lightonai/LightOnOCR-2-1B-ocr-soup"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="lighton_ocr_2_1b_ocr_soup_gguf",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        self.processor = AutoProcessor.from_pretrained(self._BASE_PROCESSOR_MODEL)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        _refresh_gguf_detection()
        pretrained_model_name = self._variant_config.pretrained_model_name
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self.processor is None:
            self._load_processor()

        # Load the full VL model from the base repo to get the vision encoder.
        # The GGUF checkpoint only has the text backbone (qwen3 architecture) so
        # we cannot load a VL model from it directly.
        vl_model = LightOnOcrForConditionalGeneration.from_pretrained(
            self._BASE_PROCESSOR_MODEL,
            torch_dtype=dtype,
        )

        # Load the GGUF text decoder as Qwen3ForCausalLM.  The GGUF arch tag is
        # "qwen3" which is supported by AutoModelForCausalLM and get_gguf_hf_weights_map.
        gguf_lm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            gguf_file=self.GGUF_FILE,
            torch_dtype=dtype,
        )

        # Transplant GGUF text-backbone weights into the VL model's language_model.
        # Keys match: Qwen3Model.state_dict() == LightOnOcrModel.language_model.state_dict()
        vl_model.model.language_model.load_state_dict(gguf_lm.model.state_dict())

        vl_model.eval()
        return vl_model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

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
