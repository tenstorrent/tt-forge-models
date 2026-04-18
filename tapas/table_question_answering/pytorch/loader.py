# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TAPAS model loader implementation for table question answering.
"""

import torch
import torch.nn as nn
import pandas as pd
from transformers import TapasConfig, TapasForQuestionAnswering, TapasTokenizer
from transformers.models.tapas.modeling_tapas import compute_token_logits
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


class TapasCompilableWrapper(nn.Module):
    """Wrapper that avoids data-dependent segment operations incompatible with torch.compile."""

    def __init__(self, tapas_qa_model):
        super().__init__()
        self.tapas = tapas_qa_model.tapas
        self.dropout = tapas_qa_model.dropout
        self.output_weights = tapas_qa_model.output_weights
        self.output_bias = tapas_qa_model.output_bias
        self.config = tapas_qa_model.config
        if hasattr(tapas_qa_model, "aggregation_classifier"):
            self.aggregation_classifier = tapas_qa_model.aggregation_classifier

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
    ):
        outputs = self.tapas(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=False,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        logits = compute_token_logits(
            sequence_output,
            self.config.temperature,
            self.output_weights,
            self.output_bias,
        )

        logits_aggregation = None
        if self.config.num_aggregation_labels > 0:
            logits_aggregation = self.aggregation_classifier(pooled_output)

        return logits, logits_aggregation


class ModelVariant(StrEnum):
    """Available TAPAS model variants for table question answering."""

    GOOGLE_TAPAS_BASE_FINETUNED_WTQ = "google/tapas-base-finetuned-wtq"


class ModelLoader(ForgeModel):
    """TAPAS model loader implementation for table question answering tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.GOOGLE_TAPAS_BASE_FINETUNED_WTQ: LLMModelConfig(
            pretrained_model_name="google/tapas-base-finetuned-wtq",
            max_length=512,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GOOGLE_TAPAS_BASE_FINETUNED_WTQ

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

        # Sample table data
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
            }
        )
        self.question = "How many goals did Cristiano Ronaldo score?"

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
        # Initialize tokenizer
        self.tokenizer = TapasTokenizer.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        qa_model = TapasForQuestionAnswering.from_pretrained(
            self.model_name, **model_kwargs
        )
        model = TapasCompilableWrapper(qa_model)
        model.eval()
        return model

    def _precompute_position_ids(self, inputs, config):
        """Pre-compute position_ids on CPU to avoid data-dependent ops at runtime.

        Replicates the logic in TapasEmbeddings.forward that uses scatter_reduce
        with dynamic shapes, which is incompatible with torch.compile backends.
        """
        from transformers.models.tapas.modeling_tapas import (
            IndexMap,
            ProductIndexMap,
            reduce_min,
            gather,
        )

        token_type_ids = inputs["token_type_ids"]
        input_shape = inputs["input_ids"].size()
        seq_length = input_shape[1]

        position_ids = torch.arange(seq_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if config.reset_position_index_per_cell:
            col_index = IndexMap(
                token_type_ids[:, :, 1], config.type_vocab_sizes[1], batch_dims=1
            )
            row_index = IndexMap(
                token_type_ids[:, :, 2], config.type_vocab_sizes[2], batch_dims=1
            )
            full_index = ProductIndexMap(col_index, row_index)
            first_position_per_segment = reduce_min(position_ids, full_index)[0]
            first_position = gather(first_position_per_segment, full_index)
            position = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
            position_ids = torch.min(
                torch.as_tensor(config.max_position_embeddings - 1),
                position - first_position,
            )

        return position_ids

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            table=self.table,
            queries=[self.question],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        config = TapasConfig.from_pretrained(self.model_name)
        inputs["position_ids"] = self._precompute_position_ids(inputs, config)

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
        aggregation = id2aggregation.get(predicted_aggregation_indices[0], "NONE")

        coordinates = predicted_answer_coordinates[0]
        if coordinates:
            cell_values = [self.table.iat[coord] for coord in coordinates]
            answer = ", ".join(cell_values)
            if aggregation != "NONE":
                answer = f"{aggregation}({answer})"
        else:
            answer = "No answer found"

        print(f"Question: {self.question}")
        print(f"Predicted answer: {answer}")
