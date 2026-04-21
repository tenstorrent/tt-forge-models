# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GROVER model loader implementation for masked language modeling on DNA sequences.
"""

from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import Optional

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
    """Available GROVER model variants for masked language modeling."""

    GROVER = "PoetschLab/GROVER"


class ModelLoader(ForgeModel):
    """GROVER model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.GROVER: LLMModelConfig(
            pretrained_model_name="PoetschLab/GROVER",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GROVER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="GROVER",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        # GROVER uses Byte Pair Encoding (BPE) on raw DNA sequences.
        # The [MASK] token marks the position to be predicted by the model.
        dna_sequence = (
            "ACTGACTGACTGACTGACTGACTGACTGACTG"
            "ACTGACTG[MASK]ACTGACTGACTGACTGACTG"
            "ACTGACTGACTGACTGACTGACTGACTGACTG"
        )

        max_length = self._variant_config.max_length
        inputs = self.tokenizer(
            dna_sequence,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_token = self.tokenizer.decode(predicted_token_id)
        return predicted_token
