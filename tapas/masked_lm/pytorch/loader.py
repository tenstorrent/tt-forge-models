# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TAPAS model loader implementation for masked language modeling.
"""

import pandas as pd
from transformers import TapasForMaskedLM, TapasTokenizer
from typing import Optional

from third_party.tt_forge_models.base import ForgeModel
from third_party.tt_forge_models.config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class _TapasDataFrame(pd.DataFrame):
    """DataFrame subclass that patches iterrows() for pandas 3.0 compatibility.

    The TAPAS tokenizer uses row[int] on Series from iterrows(), which raised
    KeyError in pandas 3.0 when the index is string column labels. Resetting
    each row's index to RangeIndex restores the pre-3.0 positional behaviour.
    """

    def iterrows(self):
        for idx, row in super().iterrows():
            yield idx, row.reset_index(drop=True)

    @property
    def _constructor(self):
        return _TapasDataFrame


class ModelVariant(StrEnum):
    """Available TAPAS model variants for masked language modeling."""

    GOOGLE_TAPAS_BASE = "google/tapas-base"


class ModelLoader(ForgeModel):
    """TAPAS model loader implementation for masked language modeling tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.GOOGLE_TAPAS_BASE: LLMModelConfig(
            pretrained_model_name="google/tapas-base",
            max_length=512,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GOOGLE_TAPAS_BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

        # Sample table data
        # Wrap in _TapasDataFrame to work around two pandas 3.0 incompatibilities
        # in the TAPAS tokenizer:
        # 1. Arrow-backed string columns reject Cell object assignment (dtype=object fixes this).
        # 2. row[int] on a string-indexed Series raises KeyError; overriding iterrows() to
        #    return integer-indexed rows restores the pre-3.0 positional-fallback behavior.
        self.table = _TapasDataFrame(
            {
                "Player": [
                    "Lionel Messi",
                    "Cristiano Ronaldo",
                    "Neymar Jr",
                    "Kylian Mbappe",
                ],
                "Goals": ["672", "700", "350", "210"],
                "Assists": ["303", "220", "240", "100"],
                "Team": ["Inter Miami", "Al Nassr", "Al Hilal", "PSG"],
            },
            dtype=object,
        )
        self.query = "Lionel Messi plays for [MASK] Miami."

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="TAPAS",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        # Initialize tokenizer
        self.tokenizer = TapasTokenizer.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = TapasForMaskedLM.from_pretrained(self.model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            table=self.table,
            queries=[self.query],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        inputs = self.load_inputs()
        logits = co_out[0]
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_token = self.tokenizer.decode(predicted_token_id)
        print("The predicted token for the [MASK] is:", predicted_token)
