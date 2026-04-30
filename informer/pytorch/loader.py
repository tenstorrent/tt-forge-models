# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Informer model loader implementation for time series forecasting.
"""

import types
import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass

from transformers import InformerForPrediction
from transformers.modeling_outputs import BaseModelOutput
from transformers.masking_utils import create_bidirectional_mask

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


@dataclass
class InformerModelConfig(ModelConfig):
    context_length: int = 24
    prediction_length: int = 24
    lags_sequence_length: int = 37
    num_time_features: int = 2
    num_static_categorical_features: int = 1


def _patch_informer_encoder(encoder):
    """Patch InformerEncoder.forward to trim the attention_mask after each distil conv layer.

    In transformers 5.x, create_bidirectional_mask returns None in eager mode (because
    _ignore_bidirectional_mask_sdpa short-circuits), but returns a real 4D tensor during
    torch.compile tracing (is_tracing() prevents the short-circuit). The encoder loop
    never updates the mask after each conv layer halves the sequence, so the second encoder
    layer sees a (1,1,24,24) mask when it expects (1,1,12,12).
    """

    def _forward(
        self,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.value_embedding(inputs_embeds)
        embed_pos = self.embed_positions(inputs_embeds.size())
        hidden_states = self.layernorm_embedding(hidden_states + embed_pos)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for idx, (encoder_layer, conv_layer) in enumerate(
            zip(self.layers, self.conv_layers)
        ):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )
                if conv_layer is not None:
                    output = conv_layer(layer_outputs[0])
                    layer_outputs = (output,) + layer_outputs[1:]
                hidden_states = layer_outputs[0]
                # Trim the 4D attention mask to the new (shorter) sequence length
                # so subsequent encoder layers see a correctly-shaped mask.
                if conv_layer is not None and attention_mask is not None:
                    new_len = hidden_states.size(1)
                    attention_mask = attention_mask[:, :, :new_len, :new_len]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, encoder_states, all_attentions] if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )

    encoder.forward = types.MethodType(_forward, encoder)


class ModelVariant(StrEnum):
    TOURISM_MONTHLY = "tourism_monthly"


class ModelLoader(ForgeModel):
    """Informer model loader for time series forecasting.

    Loads the Informer encoder-decoder transformer model with
    ProbSparse self-attention for long sequence time series forecasting.
    """

    _VARIANTS = {
        ModelVariant.TOURISM_MONTHLY: InformerModelConfig(
            pretrained_model_name="huggingface/informer-tourism-monthly",
            context_length=24,
            prediction_length=24,
            lags_sequence_length=37,
            num_time_features=2,
            num_static_categorical_features=1,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TOURISM_MONTHLY

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Informer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Informer model for time series forecasting.

        Returns:
            torch.nn.Module: The InformerForPrediction model instance.
        """
        cfg = self._variant_config

        model = InformerForPrediction.from_pretrained(
            cfg.pretrained_model_name,
            torch_dtype=dtype_override or torch.float32,
        )
        model.eval()

        _patch_informer_encoder(model.model.encoder)

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the Informer model.

        Returns:
            dict: Input tensors matching InformerForPrediction.forward signature.
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)

        # Sequence length = context_length + max(lags_sequence)
        sequence_length = cfg.context_length + cfg.lags_sequence_length

        # past_values: (batch, sequence_length)
        past_values = torch.randn(1, sequence_length, dtype=dtype)

        # past_time_features: (batch, sequence_length, num_time_features)
        past_time_features = torch.randn(
            1, sequence_length, cfg.num_time_features, dtype=dtype
        )

        # past_observed_mask: (batch, sequence_length)
        past_observed_mask = torch.ones(1, sequence_length, dtype=dtype)

        # static_categorical_features: (batch, num_static_categorical_features)
        static_categorical_features = torch.zeros(
            1, cfg.num_static_categorical_features, dtype=torch.long
        )

        # future_time_features: (batch, prediction_length, num_time_features)
        future_time_features = torch.randn(
            1, cfg.prediction_length, cfg.num_time_features, dtype=dtype
        )

        return {
            "past_values": past_values,
            "past_time_features": past_time_features,
            "past_observed_mask": past_observed_mask,
            "static_categorical_features": static_categorical_features,
            "future_time_features": future_time_features,
        }
