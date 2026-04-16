# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MMS TTS model loader implementation for text-to-speech tasks using VITS architecture.
"""

import math
from typing import Optional

import torch
from torch import nn
from transformers import AutoTokenizer, VitsModel
from transformers.models.vits import modeling_vits

# Monkey-patch _unconstrained_rational_quadratic_spline to use math.log/math.exp
# instead of np.log/np.exp, which conflicts with TorchFunctionMode during Dynamo tracing.


def _patched_unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    reverse=False,
    tail_bound=5.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    log_abs_det = torch.zeros_like(inputs)
    constant = math.log(math.exp(1 - min_derivative) - 1)

    unnormalized_derivatives = nn.functional.pad(unnormalized_derivatives, pad=(1, 1))
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    log_abs_det[outside_interval_mask] = 0.0

    (
        outputs[inside_interval_mask],
        log_abs_det[inside_interval_mask],
    ) = modeling_vits._rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        reverse=reverse,
        tail_bound=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    return outputs, log_abs_det


modeling_vits._unconstrained_rational_quadratic_spline = (
    _patched_unconstrained_rational_quadratic_spline
)

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available MMS TTS model variants."""

    BENGALI = "Bengali"
    KINYARWANDA = "Kinyarwanda"
    TELUGU = "Telugu"


class ModelLoader(ForgeModel):
    """MMS TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.BENGALI: ModelConfig(
            pretrained_model_name="facebook/mms-tts-ben",
        ),
        ModelVariant.KINYARWANDA: ModelConfig(
            pretrained_model_name="facebook/mms-tts-kin",
        ),
        ModelVariant.TELUGU: ModelConfig(
            pretrained_model_name="facebook/mms-tts-tel",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KINYARWANDA

    _SAMPLE_TEXTS = {
        ModelVariant.BENGALI: "আমাদের সিস্টেম ব্যবহার করার জন্য স্বাগতম।",
        ModelVariant.KINYARWANDA: "Muraho, murakaza neza mu gukoresha sisitemu yacu.",
        ModelVariant.TELUGU: "మా వ్యవస్థను ఉపయోగించినందుకు స్వాగతం.",
    }

    sample_text = _SAMPLE_TEXTS[DEFAULT_VARIANT]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MMS_TTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = VitsModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        sample_text = self._SAMPLE_TEXTS.get(self._variant, self.sample_text)
        inputs = self._tokenizer(sample_text, return_tensors="pt")

        return inputs
