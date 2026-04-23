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

import dataclasses
from typing import Optional, Union

import torch
from torch import nn
from transformers import AutoTokenizer
from transformers.modeling_outputs import ModelOutput
from transformers.models.mt5.modeling_mt5 import (
    MT5Config,
    MT5ForConditionalGeneration,
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


class MT5ForRegression(MT5ForConditionalGeneration):
    """MT5 model for regression (adapted from google-research/metricx).

    Subclasses MT5ForConditionalGeneration and overrides forward to:
    - Auto-create a single-step decoder input (token id 0)
    - Extract prediction from lm_logits[:, 0, 250089] (<extra_id_10>)
    - Clamp output to [0, 25]
    """

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        # Transformers 5.x looks up cls.__module__ in sys.modules to find the
        # source file and check for @use_experts_implementation. Under dynamic
        # loading the module is registered under a different key, causing a
        # KeyError. Since MT5 does not use experts_implementation, return False
        # to match the normal fallback-to-eager behavior.
        return False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[tuple, MT5ForRegressionOutput]:
        batch_size = input_ids.size(0)
        decoder_input_ids = torch.zeros(
            batch_size, 1, dtype=torch.long, device=input_ids.device
        )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            **kwargs,
        )

        lm_logits = outputs.logits

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
