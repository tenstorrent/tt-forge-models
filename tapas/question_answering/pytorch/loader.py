# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TAPAS model loader implementation for table question answering
"""
import pandas as pd
from transformers import TapasForQuestionAnswering, TapasTokenizer
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
    """Patches TAPAS tokenizer functions for pandas 3.0 compatibility."""
    import transformers.models.tapas.tokenization_tapas as tapas_mod

    if getattr(tapas_mod, "_pandas3_patched", False):
        return
    tapas_mod._pandas3_patched = True

    _orig_get_column_values = tapas_mod._get_column_values

    def _patched_add_numeric_table_values(
        table, min_consolidation_fraction=0.7, debug_info=None
    ):
        table = table.copy().astype(object)
        tapas_mod.filter_invalid_unicode_from_table(table)
        for row_index, row in table.iterrows():
            for col_index, cell in enumerate(row):
                table.iloc[row_index, col_index] = tapas_mod.Cell(text=cell)
        for col_index, column in enumerate(table.columns):
            column_values = tapas_mod._consolidate_numeric_values(
                tapas_mod._get_column_values(table, col_index),
                min_consolidation_fraction=min_consolidation_fraction,
                debug_info=(debug_info, column),
            )
            for row_index, numeric_value in column_values.items():
                table.iloc[row_index, col_index].numeric_value = numeric_value
        return table

    def _patched_get_column_values(table, col_index):
        index_to_values = {}
        for row_index, row in table.iterrows():
            text = tapas_mod.normalize_for_match(row.iloc[col_index].text)
            index_to_values[row_index] = list(tapas_mod._get_numeric_values(text))
        return index_to_values

    def _patched_class_get_column_values(self, table, col_index):
        table_numeric_values = {}
        for row_index, row in table.iterrows():
            cell = row.iloc[col_index]
            if cell.numeric_value is not None:
                table_numeric_values[row_index] = cell.numeric_value
        return table_numeric_values

    tapas_mod.add_numeric_table_values = _patched_add_numeric_table_values
    tapas_mod._get_column_values = _patched_get_column_values
    tapas_mod.TapasTokenizer._get_column_values = _patched_class_get_column_values


class ModelVariant(StrEnum):
    """Available TAPAS model variants for table question answering."""

    GOOGLE_TAPAS_BASE_FINETUNED_WTQ = "google-tapas-base-finetuned-wtq"
    LYSANDRE_TINY_TAPAS_RANDOM_SQA = "lysandre-tiny-tapas-random-sqa"


class ModelLoader(ForgeModel):
    """TAPAS model loader implementation for table question answering tasks."""

    _VARIANTS = {
        ModelVariant.GOOGLE_TAPAS_BASE_FINETUNED_WTQ: LLMModelConfig(
            pretrained_model_name="google/tapas-base-finetuned-wtq",
            max_length=512,
        ),
        ModelVariant.LYSANDRE_TINY_TAPAS_RANDOM_SQA: LLMModelConfig(
            pretrained_model_name="lysandre/tiny-tapas-random-sqa",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GOOGLE_TAPAS_BASE_FINETUNED_WTQ

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

        self.table = pd.DataFrame(
            {
                "Player": [
                    "Lionel Messi",
                    "Cristiano Ronaldo",
                    "Neymar Jr",
                    "Kylian Mbappe",
                ],
                "Goals": ["672", "680", "310", "210"],
                "Assists": ["305", "220", "200", "105"],
                "Team": [
                    "Inter Miami",
                    "Al Nassr",
                    "Al Hilal",
                    "Real Madrid",
                ],
            }
        )
        self.queries = ["How many goals does Lionel Messi have?"]

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="TAPAS",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = TapasTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = TapasForQuestionAnswering.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        _patch_tapas_tokenizer_for_pandas3()

        inputs = self.tokenizer(
            table=self.table,
            queries=self.queries,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        inputs = self.load_inputs()
        logits = co_out[0]
        logits_aggregation = co_out[1]

        (
            predicted_answer_coordinates,
            predicted_aggregation_indices,
        ) = self.tokenizer.convert_logits_to_predictions(
            inputs, logits, logits_aggregation
        )

        id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
        for coordinates, agg_index in zip(
            predicted_answer_coordinates, predicted_aggregation_indices
        ):
            aggregation = id2aggregation.get(agg_index, "NONE")
            cell_values = [self.table.iat[coord] for coord in coordinates]

            if aggregation == "NONE":
                answer = ", ".join(cell_values)
            else:
                answer = f"{aggregation} of {', '.join(cell_values)}"

            print(f"Predicted answer: {answer}")
