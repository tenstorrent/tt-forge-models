# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RNAErnie model loader implementation for masked language modeling on RNA sequences.
"""
import sys
import types

from transformers import AutoModelForMaskedLM
from typing import Optional


def _get_rna_tokenizer_class():
    # multimolecule.__init__ imports from models which has a broken transformers
    # dependency (check_model_inputs missing in transformers>=5.x). We bypass it
    # by registering a stub module so only the tokenisers subpackage is loaded.
    if "multimolecule" not in sys.modules:
        import importlib.util

        spec = importlib.util.find_spec("multimolecule")
        if spec is not None:
            stub = types.ModuleType("multimolecule")
            stub.__path__ = list(spec.submodule_search_locations)
            stub.__package__ = "multimolecule"
            sys.modules["multimolecule"] = stub

    from multimolecule.tokenisers.rna.tokenization_rna import RnaTokenizer

    return RnaTokenizer


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
    """Available RNAErnie model variants."""

    RNAERNIE = "multimolecule/rnaernie"


class ModelLoader(ForgeModel):
    """RNAErnie model loader implementation for masked language modeling on RNA sequences."""

    _VARIANTS = {
        ModelVariant.RNAERNIE: ModelConfig(
            pretrained_model_name="multimolecule/rnaernie",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RNAERNIE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="RNAErnie",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        RnaTokenizer = _get_rna_tokenizer_class()
        self.tokenizer = RnaTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        # RNA sequence with <mask> token for masked LM task
        masked_sequence = "gguc<mask>cucugguuagaccagaucugagccu"

        inputs = self.tokenizer(
            masked_sequence,
            return_tensors="pt",
            add_special_tokens=True,
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_tokens = self.tokenizer.decode(predicted_token_id)

        return predicted_tokens
