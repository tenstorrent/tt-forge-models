# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chinese ELECTRA model loader implementation for discriminator (pre-training) task.
"""

from typing import Optional

from transformers import AutoTokenizer, ElectraForPreTraining

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
    """Available Chinese ELECTRA discriminator model variants."""

    LARGE_180G = "180g-large"


class ModelLoader(ForgeModel):
    """Chinese ELECTRA model loader implementation for discriminator (pre-training) task."""

    _VARIANTS = {
        ModelVariant.LARGE_180G: LLMModelConfig(
            pretrained_model_name="hfl/chinese-electra-180g-large-discriminator",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_180G

    sample_text = "今天天气非常好，我们一起去公园散步吧。"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Chinese-ELECTRA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = ElectraForPreTraining.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model.eval()
        return self.model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        import torch

        predictions = torch.round((torch.sign(co_out[0]) + 1) / 2)
        tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer(self.sample_text, return_tensors="pt")["input_ids"][0]
        )
        for token, pred in zip(tokens, predictions[0].int().tolist()):
            label = "fake" if pred == 1 else "real"
            print(f"{token}: {label}")
