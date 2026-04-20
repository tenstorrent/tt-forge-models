# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERTic model loader implementation for discriminator (pre-training) task.

classla/bcms-bertic is an ELECTRA-architecture transformer trained on more than
8 billion tokens of Bosnian, Croatian, Montenegrin and Serbian text.
"""

import torch
from transformers import ElectraForPreTraining, ElectraTokenizerFast

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
    """Available BERTic discriminator model variants."""

    BCMS_BERTIC = "bcms_bertic"


class ModelLoader(ForgeModel):
    """BERTic model loader implementation for discriminator (pre-training) task."""

    _VARIANTS = {
        ModelVariant.BCMS_BERTIC: LLMModelConfig(
            pretrained_model_name="classla/bcms-bertic",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BCMS_BERTIC

    sample_text = "Brza smeđa lisica skače preko lijenog psa"

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant=None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BERTic",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = ElectraTokenizerFast.from_pretrained(self.model_name)

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
        predictions = torch.round((torch.sign(co_out[0]) + 1) / 2)
        tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer(self.sample_text, return_tensors="pt",)[
                "input_ids"
            ][0]
        )
        for token, pred in zip(tokens, predictions[0].int().tolist()):
            label = "fake" if pred == 1 else "real"
            print(f"{token}: {label}")
