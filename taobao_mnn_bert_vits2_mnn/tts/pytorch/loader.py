# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Taobao-MNN BERT-VITS2-MNN model loader implementation for text-to-speech tasks.

This is an MNN (Mobile Neural Network) converted int8 quantized BERT-VITS2
text-to-speech model exported for Alibaba's MNN inference framework.
"""

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
    """Available BERT-VITS2-MNN model variants for text-to-speech."""

    BERT_VITS2_MNN = "bert_vits2_mnn"


class ModelLoader(ForgeModel):
    """Taobao-MNN BERT-VITS2-MNN model loader for text-to-speech."""

    _VARIANTS = {
        ModelVariant.BERT_VITS2_MNN: ModelConfig(
            pretrained_model_name="taobao-mnn/bert-vits2-MNN",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BERT_VITS2_MNN

    sample_text = "Hello, this is a test of text to speech."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="taobao_mnn_bert_vits2_mnn",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        from transformers import AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
            )

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="pt",
        )
        return inputs
