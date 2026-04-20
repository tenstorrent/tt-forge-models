# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pix2Struct model loader implementation for document question answering using PyTorch.
"""

import torch
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
    """Available Pix2Struct PyTorch document question answering model variants."""

    DOCVQA_BASE = "DocVQA_Base"


class ModelLoader(ForgeModel):
    """Pix2Struct model loader implementation for document question answering (PyTorch)."""

    _VARIANTS = {
        ModelVariant.DOCVQA_BASE: ModelConfig(
            pretrained_model_name="google/pix2struct-docvqa-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DOCVQA_BASE

    sample_question = "What is the title of this paper?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Pix2Struct",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import Pix2StructProcessor

        self._processor = Pix2StructProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import Pix2StructForConditionalGeneration

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Pix2StructForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        from huggingface_hub import hf_hub_download
        from PIL import Image

        if self._processor is None:
            self._load_processor()

        image_path = hf_hub_download(
            repo_id="hf-internal-testing/fixtures_docvqa",
            filename="nougat_paper.png",
            repo_type="dataset",
        )
        image = Image.open(image_path).convert("RGB")

        inputs = self._processor(
            images=image,
            text=self.sample_question,
            return_tensors="pt",
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].is_floating_point():
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs

    def decode_output(self, co_out):
        if self._processor is None:
            self._load_processor()

        generated_text = self._processor.batch_decode(co_out, skip_special_tokens=True)[
            0
        ]
        print(f"Generated text: {generated_text}")
        return generated_text
