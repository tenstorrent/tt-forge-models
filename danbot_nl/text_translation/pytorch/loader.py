# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DanbotNL model loader implementation for natural language to Danbooru tag translation.
"""

from typing import Optional

from transformers import AutoModelForPreTraining, AutoProcessor

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available DanbotNL model variants for text translation."""

    DANBOT_NL_2408_260M = "DanbotNL_2408_260M"


class ModelLoader(ForgeModel):
    """DanbotNL model loader implementation for text translation."""

    _VARIANTS = {
        ModelVariant.DANBOT_NL_2408_260M: LLMModelConfig(
            pretrained_model_name="dartags/DanbotNL-2408-260M",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DANBOT_NL_2408_260M

    sample_text = "一人の猫耳の少女が座ってこっちを見ている。"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DanbotNL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TRANSLATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForPreTraining.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        decoder_text = self.processor.decoder_tokenizer.apply_chat_template(
            {
                "aspect_ratio": "tall",
                "rating": "general",
                "length": "very_short",
                "translate_mode": "exact",
            },
            tokenize=False,
        )

        inputs = self.processor(
            encoder_text=self.sample_text,
            decoder_text=decoder_text,
            return_tensors="pt",
        )

        return inputs
