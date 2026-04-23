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


def _patch_tapas_tokenizer_for_pandas3():
    """Patch TAPAS tokenizer's _get_column_values to use iloc for pandas 3.0.

    pandas 3.0 removed the positional fallback for Series.__getitem__ with
    integer keys on a string-indexed Series.  The TAPAS tokenizer uses
    row[col_index] (integer) on rows from iterrows() whose index is the
    string column names, causing KeyError.  Two functions need patching:
    the module-level _get_column_values and the TapasTokenizer instance method.
    Patching once per process.
    """
    import transformers.models.tapas.tokenization_tapas as _tapas
    from transformers import TapasTokenizer

    if getattr(_tapas, "_patched_for_pandas3", False):
        return

    _normalize = _tapas.normalize_for_match
    _get_numeric = _tapas._get_numeric_values

    # Patch module-level function used by add_numeric_table_values
    def _patched_module_get_column_values(table, col_index):
        index_to_values = {}
        for row_index, row in table.iterrows():
            text = _normalize(row.iloc[col_index].text)
            index_to_values[row_index] = list(_get_numeric(text))
        return index_to_values

    _tapas._get_column_values = _patched_module_get_column_values

    # Patch instance method used by _get_numeric_column_ranks
    def _patched_instance_get_column_values(self, table, col_index):
        table_numeric_values = {}
        for row_index, row in table.iterrows():
            cell = row.iloc[col_index]
            if cell.numeric_value is not None:
                table_numeric_values[row_index] = cell.numeric_value
        return table_numeric_values

    TapasTokenizer._get_column_values = _patched_instance_get_column_values

    _tapas._patched_for_pandas3 = True


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

        # dtype=object avoids the pandas 3.0 Arrow-backed string backend
        # which rejects Cell object assignment inside the TAPAS tokenizer.
        self.table = pd.DataFrame(
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

        _patch_tapas_tokenizer_for_pandas3()

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
