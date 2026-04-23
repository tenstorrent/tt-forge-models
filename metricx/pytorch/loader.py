# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MetricX-24 model loader implementation for translation quality evaluation.

MetricX-24 is a learned regression metric for machine translation quality
based on mT5. It outputs a score in [0, 25] (lower is better, MQM convention).

The MT5ForRegression class is adapted from the metricx project
(https://github.com/google-research/metricx), Apache-2.0 License.

Available variants:
- HYBRID_XL_V2P6: google/metricx-24-hybrid-xl-v2p6-bfloat16
- HYBRID_XXL_V2P6: google/metricx-24-hybrid-xxl-v2p6-bfloat16
- HYBRID_LARGE_V2P6_FP32: google/metricx-24-hybrid-large-v2p6
"""

import copy
import dataclasses
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.models.mt5.modeling_mt5 import (
    MT5Config,
    MT5PreTrainedModel,
    MT5Stack,
    __HEAD_MASK_WARNING_MSG,
)

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


@dataclasses.dataclass
class MT5ForRegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    predictions: torch.FloatTensor = None


class MT5ForRegression(MT5PreTrainedModel):
    """MT5 model for regression (adapted from google-research/metricx)."""

    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.post_init()

        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], MT5ForRegressionOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        batch_size = input_ids.size(0)
        decoder_input_ids = torch.LongTensor([0]).repeat(batch_size).reshape(-1, 1)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        # 250089 = <extra_id_10>
        predictions = lm_logits[:, 0, 250089]
        predictions = torch.clamp(predictions, 0, 25)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            labels = labels.to(predictions.device)
            loss = loss_fct(predictions.view(-1), labels.view(-1))

        return MT5ForRegressionOutput(
            loss=loss,
            predictions=predictions,
        )


class ModelVariant(StrEnum):
    """Available MetricX-24 model variants."""

    HYBRID_XL_V2P6 = "Hybrid_XL_v2p6"
    HYBRID_XXL_V2P6 = "Hybrid_XXL_v2p6"
    HYBRID_LARGE_V2P6_FP32 = "Hybrid_Large_v2p6_fp32"


class ModelLoader(ForgeModel):
    """MetricX-24 model loader for translation quality regression."""

    _VARIANTS = {
        ModelVariant.HYBRID_XL_V2P6: LLMModelConfig(
            pretrained_model_name="google/metricx-24-hybrid-xl-v2p6-bfloat16",
            max_length=512,
        ),
        ModelVariant.HYBRID_XXL_V2P6: LLMModelConfig(
            pretrained_model_name="google/metricx-24-hybrid-xxl-v2p6-bfloat16",
            max_length=512,
        ),
        ModelVariant.HYBRID_LARGE_V2P6_FP32: LLMModelConfig(
            pretrained_model_name="google/metricx-24-hybrid-large-v2p6",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HYBRID_XXL_V2P6

    # Sample input: reference-based translation quality evaluation
    sample_source = "The quick brown fox jumps over the lazy dog."
    sample_translation = "Le rapide renard brun saute par-dessus le chien paresseux."
    sample_reference = "Le renard brun rapide saute par-dessus le chien paresseux."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MetricX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MetricX-24 MT5ForRegression model."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = MT5ForRegression.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample inputs for MetricX-24.

        Input is formatted as: "candidate: {translation} | reference: {reference}"
        for reference-based evaluation.
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        input_text = (
            f"candidate: {self.sample_translation} | "
            f"reference: {self.sample_reference}"
        )

        inputs = self.tokenizer(
            input_text,
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs
