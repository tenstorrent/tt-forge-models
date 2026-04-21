# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ruT5 Base Absum model loader for summarization.
A Russian T5 model fine-tuned for abstractive summarization,
based on cointegrated/rut5-base-multitask.
"""
from typing import Optional

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

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
    """Available ruT5 Base Absum model variants."""

    RUT5_BASE_ABSUM = "Base_Absum"


class ModelLoader(ForgeModel):
    """ruT5 Base Absum model loader for summarization."""

    _VARIANTS = {
        ModelVariant.RUT5_BASE_ABSUM: ModelConfig(
            pretrained_model_name="cointegrated/rut5-base-absum",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RUT5_BASE_ABSUM

    sample_text = (
        "Высота башни составляет 324 метра (1063 фута), примерно такая же "
        "высота, как у 81-этажного здания, и самое высокое сооружение в Париже. "
        "Его основание квадратно, размером 125 метров (410 футов) с любой "
        "стороны. Во время строительства Эйфелева башня превзошла монумент "
        "Вашингтона, став самым высоким искусственным сооружением в мире, и "
        "этот титул она удерживала в течение 41 года до завершения "
        "строительство здания Крайслер в Нью-Йорке в 1930 году."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self._cached_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ruT5 Base Absum",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_SUMMARIZATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        decoder_start_token_tensor = torch.tensor(
            self._cached_model.generation_config.decoder_start_token_id,
            dtype=torch.long,
        )
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_tensor
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
