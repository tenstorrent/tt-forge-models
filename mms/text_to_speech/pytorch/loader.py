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

# Two patches for the VITS spline functions:
#
# 1. _unconstrained_rational_quadratic_spline: replace boolean mask indexing
#    (dynamic shapes → CPU-fallback AttributeError) with torch.where on full
#    tensors, and replace np.log/np.exp with math equivalents to avoid the
#    numpy-float64 / Dynamo TorchFunctionMode interaction.
#
# 2. _rational_quadratic_spline: in the reverse path, clamp discriminant ≥ 0
#    before sqrt because clamped outside-interval elements can produce
#    (spurious) negative discriminants; torch.where in the outer function
#    discards those results anyway.


def _patched_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    reverse,
    tail_bound,
    min_bin_width,
    min_bin_height,
    min_derivative,
):
    upper_bound = tail_bound
    lower_bound = -tail_bound
    num_bins = unnormalized_widths.shape[-1]

    widths = nn.functional.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = nn.functional.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (upper_bound - lower_bound) * cumwidths + lower_bound
    cumwidths[..., 0] = lower_bound
    cumwidths[..., -1] = upper_bound
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + nn.functional.softplus(unnormalized_derivatives)

    heights = nn.functional.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = nn.functional.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (upper_bound - lower_bound) * cumheights + lower_bound
    cumheights[..., 0] = lower_bound
    cumheights[..., -1] = upper_bound
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_locations = cumheights if reverse else cumwidths
    bin_locations[..., -1] += 1e-6
    bin_idx = torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1
    bin_idx = bin_idx[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    intermediate1 = input_derivatives + input_derivatives_plus_one - 2 * input_delta
    if not reverse:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)
        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + intermediate1 * theta_one_minus_theta
        outputs = input_cumheights + numerator / denominator
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        log_abs_det = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, log_abs_det
    else:
        intermediate2 = inputs - input_cumheights
        intermediate3 = intermediate2 * intermediate1
        a = input_heights * (input_delta - input_derivatives) + intermediate3
        b = input_heights * input_derivatives - intermediate3
        c = -input_delta * intermediate2
        # Clamp discriminant ≥ 0: outside-interval elements (clamped to boundary)
        # can produce spurious negatives; torch.where in the caller discards them.
        discriminant = torch.clamp(b.pow(2) - 4 * a * c, min=0)
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + intermediate1 * theta_one_minus_theta
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        log_abs_det = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -log_abs_det


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

    # math.log/math.exp avoids numpy scalars that confuse Dynamo under TorchFunctionMode.
    constant = math.log(math.exp(1 - min_derivative) - 1)

    unnormalized_derivatives = nn.functional.pad(unnormalized_derivatives, pad=(1, 1))
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    # Clamp to valid range so the bounds check in _rational_quadratic_spline passes
    # for all elements.  torch.where below restores the identity transform
    # (output=input, log_abs_det=0) for elements outside the interval.
    clamped_inputs = inputs.clamp(-tail_bound, tail_bound)
    full_outputs, full_log_abs_det = _patched_rational_quadratic_spline(
        inputs=clamped_inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        reverse=reverse,
        tail_bound=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    outputs = torch.where(inside_interval_mask, full_outputs, inputs)
    log_abs_det = torch.where(inside_interval_mask, full_log_abs_det, torch.zeros_like(inputs))
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
