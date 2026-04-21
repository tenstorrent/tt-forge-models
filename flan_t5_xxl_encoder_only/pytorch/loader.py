# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLAN-T5 XXL Encoder-Only model loader implementation for text embedding generation.
"""

from transformers import AutoTokenizer, T5EncoderModel, AutoConfig
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available FLAN-T5 XXL Encoder-Only model variants."""

    XXL_ENCODER_ONLY = "XXL_Encoder_Only"


class ModelLoader(ForgeModel):
    """FLAN-T5 XXL Encoder-Only model loader for text embedding generation."""

    _VARIANTS = {
        ModelVariant.XXL_ENCODER_ONLY: LLMModelConfig(
            pretrained_model_name="silveroxides/flan-t5-xxl-encoder-only",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XXL_ENCODER_ONLY

    MODEL_SUBFOLDER = "text_encoder_2"
    TOKENIZER_SUBFOLDER = "tokenizer_2"

    sample_text = "A beautiful sunset over the mountains with warm golden light."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="FLAN_T5_XXL_Encoder_Only",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {"subfolder": self.TOKENIZER_SUBFOLDER}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"subfolder": self.MODEL_SUBFOLDER}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, subfolder=self.MODEL_SUBFOLDER
            )
            config.num_layers = self.num_layers
            model_kwargs["config"] = config

        model = T5EncoderModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            truncation=True,
        )

        return inputs
