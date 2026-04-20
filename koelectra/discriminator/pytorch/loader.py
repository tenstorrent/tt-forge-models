# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KoELECTRA model loader implementation for discriminator (pre-training) task.
"""

from transformers import ElectraForPreTraining, AutoTokenizer
from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available KoELECTRA discriminator model variants."""

    MONOLOGG_KOELECTRA_SMALL_V3_DISCRIMINATOR = (
        "monologg_koelectra_small_v3_discriminator"
    )


class ModelLoader(ForgeModel):
    """KoELECTRA model loader implementation for discriminator (pre-training) task."""

    _VARIANTS = {
        ModelVariant.MONOLOGG_KOELECTRA_SMALL_V3_DISCRIMINATOR: LLMModelConfig(
            pretrained_model_name="monologg/koelectra-small-v3-discriminator",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MONOLOGG_KOELECTRA_SMALL_V3_DISCRIMINATOR

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
            model="KoELECTRA",
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

        sentence = "나는 한국어 모델을 사용하여 문장을 분석하고 있습니다"
        inputs = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        import torch

        sentence = "나는 한국어 모델을 사용하여 문장을 분석하고 있습니다"
        predictions = torch.round((torch.sign(co_out[0]) + 1) / 2)
        tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer(sentence, return_tensors="pt")["input_ids"][0]
        )
        for token, pred in zip(tokens, predictions[0].int().tolist()):
            label = "fake" if pred == 1 else "real"
            print(f"{token}: {label}")
