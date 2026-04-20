# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hypencoder dual-encoder model definitions.

Vendored from https://github.com/jfkback/hypencoder-paper with training and
loss logic removed. Only the modules required to reconstruct the checkpoint
for inference are retained so the state-dict keys match the weights published
on Hugging Face.
"""
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .q_net import RepeatedDenseBlockConverter


@dataclass
class EncoderOutput(ModelOutput):
    representation: Any = None
    loss: Optional[torch.Tensor] = None


@dataclass
class DualEncoderOutput(ModelOutput):
    query_output: Optional[EncoderOutput] = None
    passage_output: Optional[EncoderOutput] = None
    similarity: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


@dataclass
class HypencoderOutput(EncoderOutput):
    embedding_representation: Optional[torch.Tensor] = None


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dim: int,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    score = torch.einsum("bqd,bkd->bqk", query, key) / math.sqrt(dim)

    if mask is not None:
        score = score.masked_fill(mask.unsqueeze(1) == 0, -float("Inf"))

    attention = F.softmax(score, -1)

    context = torch.einsum("bqk,bkd->bqd", [attention, value])
    return context, attention


class BaseDualEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        query_encoder_type: str = "",
        passage_encoder_type: str = "",
        query_encoder_kwargs: Optional[Dict] = None,
        passage_encoder_kwargs: Optional[Dict] = None,
        loss_type: Union[str, List[str]] = "",
        loss_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]] = None,
        shared_encoder: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if query_encoder_kwargs is None:
            query_encoder_kwargs = {}
        if passage_encoder_kwargs is None:
            passage_encoder_kwargs = {}
        if loss_kwargs is None:
            loss_kwargs = {}

        if isinstance(loss_type, str):
            loss_type = [loss_type]

        if isinstance(loss_kwargs, dict):
            loss_kwargs = [loss_kwargs]

        self.query_encoder_type = query_encoder_type
        self.passage_encoder_type = passage_encoder_type
        self.query_encoder_kwargs = query_encoder_kwargs
        self.passage_encoder_kwargs = passage_encoder_kwargs
        self.loss_type = loss_type
        self.loss_kwargs = loss_kwargs
        self.shared_encoder = shared_encoder


class BaseDualEncoder(PreTrainedModel):
    config_class = BaseDualEncoderConfig

    def __init__(self, config: BaseDualEncoderConfig):
        super().__init__(config)

    def post_init(self):
        # The dual encoder's sub-encoders are instantiated in subclasses after
        # super().__init__ returns, so delegate post_init to the subclass.
        pass

    def forward(
        self,
        query_input_ids: Optional[torch.LongTensor] = None,
        query_attention_mask: Optional[torch.LongTensor] = None,
        passage_input_ids: Optional[torch.LongTensor] = None,
        passage_attention_mask: Optional[torch.LongTensor] = None,
    ) -> DualEncoderOutput:
        if query_input_ids is None and passage_input_ids is None:
            raise ValueError(
                "At least one of query_input_ids or passage_input_ids must be provided"
            )

        query_output = None
        if query_input_ids is not None:
            query_output = self.query_encoder(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
            )

        passage_output = None
        if passage_input_ids is not None:
            passage_output = self.passage_encoder(
                input_ids=passage_input_ids,
                attention_mask=passage_attention_mask,
            )

        return DualEncoderOutput(
            query_output=query_output,
            passage_output=passage_output,
        )


class HypencoderConfig(PretrainedConfig):
    def __init__(
        self,
        model_name_or_path: str = "",
        freeze_transformer: bool = False,
        converter_kwargs: Optional[Dict] = None,
        embedding_representation: Optional[str] = None,
        base_encoder_output_dim: int = 768,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if converter_kwargs is None:
            converter_kwargs = {}
        self.model_name_or_path = model_name_or_path
        self.freeze_transformer = freeze_transformer
        self.converter_kwargs = converter_kwargs
        self.embedding_representation = embedding_representation
        self.base_encoder_output_dim = base_encoder_output_dim


class Hypencoder(PreTrainedModel):
    config_class = HypencoderConfig

    def __init__(self, config: HypencoderConfig) -> None:
        super().__init__(config)
        # Initialize from config (empty weights) — the outer dual-encoder
        # from_pretrained supplies the fine-tuned state dict.
        inner_config = AutoConfig.from_pretrained(config.model_name_or_path)
        self.transformer = AutoModel.from_config(inner_config)
        self.weight_to_model_converter = RepeatedDenseBlockConverter(
            **config.converter_kwargs
        )

        self.weight_shapes = self.weight_to_model_converter.weight_shapes
        self.bias_shapes = self.weight_to_model_converter.bias_shapes

        self._initialize_hyper_head()

    def _initialize_hyper_head(self) -> None:
        model_dim = self.config.base_encoder_output_dim

        self.hyper_base_matrices = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, out_dim, in_dim))
                for in_dim, out_dim in self.weight_shapes
            ]
        )

        self.hyper_base_vectors = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(out_dim, in_dim))
                for in_dim, out_dim in self.bias_shapes
            ]
        )

        self.weight_query_embeddings = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, out_dim, in_dim))
                for in_dim, out_dim in self.weight_shapes
            ]
        )

        self.bias_query_embeddings = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, out_dim, in_dim))
                for in_dim, out_dim in self.bias_shapes
            ]
        )

        self.weight_hyper_projection = nn.ParameterList(
            [nn.Linear(in_dim, in_dim) for in_dim, _ in self.weight_shapes]
        )

        self.bias_hyper_projection = nn.ParameterList(
            [nn.Linear(in_dim, in_dim) for in_dim, _ in self.bias_shapes]
        )

        self.key_projections = nn.ParameterList(
            [
                nn.Linear(model_dim, in_dim)
                for in_dim, _ in (self.weight_shapes + self.bias_shapes)
            ]
        )

        self.value_projections = nn.ParameterList(
            [
                nn.Linear(model_dim, in_dim)
                for in_dim, _ in (self.weight_shapes + self.bias_shapes)
            ]
        )

    def _get_weights_and_biases(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        batch_size = last_hidden_state.size(0)

        keys = [
            key_projection(last_hidden_state) for key_projection in self.key_projections
        ]
        values = [
            value_projection(last_hidden_state)
            for value_projection in self.value_projections
        ]

        weights = []
        for i in range(len(self.weight_shapes)):
            weights.append(
                scaled_dot_product_attention(
                    query=self.weight_query_embeddings[i].repeat_interleave(
                        batch_size, dim=0
                    ),
                    key=keys[i],
                    value=values[i],
                    dim=self.weight_shapes[i][1],
                    mask=attention_mask,
                )[0]
            )

        biases = []
        offset = len(self.weight_shapes)
        for i in range(len(self.bias_shapes)):
            biases.append(
                scaled_dot_product_attention(
                    query=self.bias_query_embeddings[i].repeat_interleave(
                        batch_size, dim=0
                    ),
                    key=keys[i + offset],
                    value=values[i + offset],
                    dim=self.bias_shapes[i][1],
                    mask=attention_mask,
                )[0]
            )

        weights_final = []
        biases_final = []

        for i in range(len(self.weight_shapes)):
            weights_final.append(
                self.weight_hyper_projection[i](
                    F.layer_norm(F.relu(weights[i]), weights[i].shape[2:])
                )
            )

        for i in range(len(self.bias_shapes)):
            biases_final.append(
                self.bias_hyper_projection[i](
                    F.layer_norm(F.relu(biases[i]), biases[i].shape[2:])
                )
            )

        weights_final = [
            (
                weights_final[i] + self.hyper_base_matrices[i].repeat(batch_size, 1, 1)
            ).transpose(2, 1)
            for i in range(len(self.weight_shapes))
        ]

        biases_final = [
            (
                biases_final[i] + self.hyper_base_vectors[i].repeat(batch_size, 1, 1)
            ).transpose(2, 1)
            for i in range(len(self.bias_shapes))
        ]

        return weights_final, biases_final

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        transformer_output = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )

        last_hidden_state = transformer_output.last_hidden_state

        matrices, vectors = self._get_weights_and_biases(
            last_hidden_state, attention_mask
        )

        models = self.weight_to_model_converter(
            matrices, vectors, is_training=self.training
        )

        output = HypencoderOutput(representation=models)

        if self.config.embedding_representation is not None:
            if self.config.embedding_representation == "mean":
                output.embedding_representation = last_hidden_state.sum(dim=1) / (
                    attention_mask.sum(dim=1, keepdim=True)
                )
            elif self.config.embedding_representation == "cls":
                output.embedding_representation = last_hidden_state[:, 0]
            else:
                raise ValueError("Unknown embedding representation type")

        return output


class TextEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        model_name_or_path: str = "",
        pooling_type: str = "cls",
        freeze_transformer: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name_or_path = model_name_or_path
        self.pooling_type = pooling_type
        self.freeze_transformer = freeze_transformer


class TextEncoder(PreTrainedModel):
    config_class = TextEncoderConfig

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__(config)
        inner_config = AutoConfig.from_pretrained(config.model_name_or_path)
        self.transformer = AutoModel.from_config(inner_config)
        self.pooling_type = config.pooling_type

        if self.pooling_type == "mean":
            self.pool = self.mean_pool
        elif self.pooling_type == "cls":
            self.pool = self.cls_pool

    def mean_pool(self, last_hidden_state, attention_mask):
        return last_hidden_state.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

    def cls_pool(self, last_hidden_state, attention_mask):
        return last_hidden_state[:, 0]

    def forward(self, input_ids, attention_mask):
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.pool(output.last_hidden_state, attention_mask)
        return EncoderOutput(representation=pooled_output)


class HypencoderDualEncoderConfig(BaseDualEncoderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class HypencoderDualEncoder(BaseDualEncoder):
    config_class = HypencoderDualEncoderConfig

    def __init__(self, config: HypencoderDualEncoderConfig):
        super().__init__(config)

        self.query_encoder = Hypencoder(HypencoderConfig(**config.query_encoder_kwargs))

        self.passage_encoder = TextEncoder(
            TextEncoderConfig(**config.passage_encoder_kwargs)
        )

        if config.shared_encoder:
            self.passage_encoder.transformer = self.query_encoder.transformer

        # Now that sub-encoders are attached, run PreTrainedModel.post_init to
        # populate tied_weights_keys metadata expected by from_pretrained.
        PreTrainedModel.post_init(self)


class HypencoderScoringWrapper(nn.Module):
    """Combines the dual encoder forward pass into a single scoring tensor.

    The query encoder emits a per-query neural network (q-net); the passage
    encoder emits a pooled embedding. Applying the q-net to the embedding
    yields a relevance score suitable as a test harness output tensor.
    """

    def __init__(self, dual_encoder: HypencoderDualEncoder):
        super().__init__()
        self.dual_encoder = dual_encoder

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        passage_input_ids: torch.Tensor,
        passage_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        q_net = self.dual_encoder.query_encoder(
            input_ids=query_input_ids, attention_mask=query_attention_mask
        ).representation

        passage_embedding = self.dual_encoder.passage_encoder(
            input_ids=passage_input_ids, attention_mask=passage_attention_mask
        ).representation

        passage_embedding = passage_embedding.unsqueeze(1)
        scores = q_net(passage_embedding)
        return scores.squeeze(-1).squeeze(-1)
