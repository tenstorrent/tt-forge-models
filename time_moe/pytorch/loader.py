# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TimeMoE model loader implementation for time series forecasting.
"""

import torch
from typing import Optional
from dataclasses import dataclass

from transformers import AutoModelForCausalLM
from transformers import DynamicCache


def _patch_dynamic_cache():
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = (
            lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
        )

    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def _from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            if past_key_values is not None:
                for layer_idx in range(len(past_key_values)):
                    key_states, value_states = past_key_values[layer_idx]
                    cache.update(key_states, value_states, layer_idx)
            return cache

        DynamicCache.from_legacy_cache = _from_legacy_cache

    if not hasattr(DynamicCache, "to_legacy_cache"):
        DynamicCache.to_legacy_cache = lambda self: tuple(
            (self.key_cache[layer_idx], self.value_cache[layer_idx])
            for layer_idx in range(len(self))
        )


_patch_dynamic_cache()

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
class TimeMoEConfig(ModelConfig):
    context_length: int = 512
    prediction_length: int = 96


class ModelVariant(StrEnum):
    BASE_200M = "base-200m"


class ModelLoader(ForgeModel):
    """TimeMoE model loader for time series forecasting.

    Loads the TimeMoE Mixture of Experts causal model for
    zero-shot time series forecasting.
    """

    _VARIANTS = {
        ModelVariant.BASE_200M: TimeMoEConfig(
            pretrained_model_name="Maple728/TimeMoE-200M",
            context_length=512,
            prediction_length=96,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_200M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="TimeMoE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the TimeMoE model for time series forecasting.

        Returns:
            torch.nn.Module: The TimeMoE causal LM model instance.
        """
        cfg = self._variant_config

        model = AutoModelForCausalLM.from_pretrained(
            cfg.pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=dtype_override or torch.float32,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample time series inputs for the model.

        Returns:
            torch.Tensor: Input tensor of shape (batch_size, context_length).
        """
        cfg = self._variant_config
        dtype = dtype_override or torch.float32

        torch.manual_seed(42)
        inputs = torch.randn(1, cfg.context_length, dtype=dtype)

        return inputs
