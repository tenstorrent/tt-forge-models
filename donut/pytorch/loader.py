# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Donut (Document Understanding Transformer) model loader for DocVQA.

Donut is a vision-encoder-decoder model: DonutSwin (vision) + MBart (text).
Loaded as a VisionEncoderDecoderModel; forward returns Seq2SeqLMOutput
(default `unpack_forward_output` handler resolves to `.logits`).
"""
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available Donut model variants."""

    BASE_FINETUNED_DOCVQA = "Base_Finetuned_DocVQA"


class ModelLoader(ForgeModel):
    """Donut model loader for document understanding (DocVQA)."""

    _VARIANTS = {
        ModelVariant.BASE_FINETUNED_DOCVQA: ModelConfig(
            pretrained_model_name="naver-clova-ix/donut-base-finetuned-docvqa",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_FINETUNED_DOCVQA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Donut",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = DonutProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        self.model = VisionEncoderDecoderModel.from_pretrained(
            pretrained_model_name, **kwargs
        )

        if dtype_override is not None:
            self.model = self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        # Sample image — DonutProcessor will resize/pad to the encoder's
        # configured input size (2560x1920 for the docvqa checkpoint).
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        if batch_size != 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        # DocVQA task prompt format:
        #   "<s_docvqa><s_question>{q}</s_question><s_answer>"
        # For a forward-pass test we tokenize the task prompt header (without
        # any specific question) so the decoder sees a valid sequence start.
        task_prompt = "<s_docvqa>"
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        if batch_size != 1:
            decoder_input_ids = decoder_input_ids.repeat_interleave(batch_size, dim=0)

        return [pixel_values, decoder_input_ids]
