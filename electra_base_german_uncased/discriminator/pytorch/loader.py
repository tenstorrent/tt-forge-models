# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
German ELECTRA (electra-base-german-uncased) model loader implementation for discriminator (pre-training) task.
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
    """Available German ELECTRA (electra-base-german-uncased) model variants for the discriminator task."""

    ELECTRA_BASE_GERMAN_UNCASED = "german-nlp-group/electra-base-german-uncased"


class ModelLoader(ForgeModel):
    """German ELECTRA (electra-base-german-uncased) model loader implementation for discriminator (pre-training) task."""

    _VARIANTS = {
        ModelVariant.ELECTRA_BASE_GERMAN_UNCASED: LLMModelConfig(
            pretrained_model_name="german-nlp-group/electra-base-german-uncased",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ELECTRA_BASE_GERMAN_UNCASED

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = "Berlin ist die Hauptstadt von Deutschland."
        self.max_length = self._variant_config.max_length or 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="Electra_Base_German_Uncased",
            variant=variant_name,
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

        model = ElectraForPreTraining.from_pretrained(self.model_name, **model_kwargs)
        model.eval()
        return model

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
        """Decode the model output for the ELECTRA discriminator task."""
        import torch

        predictions = torch.round((torch.sign(co_out[0]) + 1) / 2)
        tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer(self.sample_text, return_tensors="pt")["input_ids"][0]
        )
        for token, pred in zip(tokens, predictions[0].int().tolist()):
            label = "fake" if pred == 1 else "real"
            print(f"{token}: {label}")
