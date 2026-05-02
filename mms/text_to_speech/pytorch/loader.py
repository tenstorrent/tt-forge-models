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


# VitsModel.forward at line 1362 calls torch.arange(predicted_lengths.max(), ...)
# where predicted_lengths is an XLA tensor.  Getting .max() requires a
# device-to-host transfer which fails on TT.  Replace with a static upper bound
# derived from the input sequence length, so the arange argument is a Python int.
_MAX_FRAMES_PER_INPUT_TOKEN = 100  # generous upper bound for TTS duration


def _patched_vits_forward(
    self,
    input_ids=None,
    attention_mask=None,
    speaker_id=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    labels=None,
    **kwargs,
):
    from transformers.models.vits.modeling_vits import VitsModelOutput

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if labels is not None:
        raise NotImplementedError("Training of VITS is not supported yet.")

    mask_dtype = self.text_encoder.embed_tokens.weight.dtype
    if attention_mask is not None:
        input_padding_mask = attention_mask.unsqueeze(-1).to(mask_dtype)
    else:
        input_padding_mask = torch.ones_like(input_ids).unsqueeze(-1).to(mask_dtype)

    if self.config.num_speakers > 1 and speaker_id is not None:
        if not 0 <= speaker_id < self.config.num_speakers:
            raise ValueError(f"Set `speaker_id` in the range 0-{self.config.num_speakers - 1}.")
        if isinstance(speaker_id, int):
            speaker_id = torch.full(size=(1,), fill_value=speaker_id, device=self.device)
        speaker_embeddings = self.embed_speaker(speaker_id).unsqueeze(-1)
    else:
        speaker_embeddings = None

    text_encoder_output = self.text_encoder(
        input_ids=input_ids,
        padding_mask=input_padding_mask,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = text_encoder_output[0] if not return_dict else text_encoder_output.last_hidden_state
    hidden_states = hidden_states.transpose(1, 2)
    input_padding_mask = input_padding_mask.transpose(1, 2)
    prior_means = text_encoder_output[1] if not return_dict else text_encoder_output.prior_means
    prior_log_variances = text_encoder_output[2] if not return_dict else text_encoder_output.prior_log_variances

    if self.config.use_stochastic_duration_prediction:
        log_duration = self.duration_predictor(
            hidden_states,
            input_padding_mask,
            speaker_embeddings,
            reverse=True,
            noise_scale=self.noise_scale_duration,
        )
    else:
        log_duration = self.duration_predictor(hidden_states, input_padding_mask, speaker_embeddings)

    length_scale = 1.0 / self.speaking_rate
    duration = torch.ceil(torch.exp(log_duration) * input_padding_mask * length_scale)
    predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()

    # Use a static upper bound derived from the input length to avoid
    # predicted_lengths.max() (device-to-host transfer).
    static_max_output_length = input_ids.shape[1] * _MAX_FRAMES_PER_INPUT_TOKEN
    indices = torch.arange(static_max_output_length, dtype=predicted_lengths.dtype, device=predicted_lengths.device)
    output_padding_mask = indices.unsqueeze(0) < predicted_lengths.unsqueeze(1)
    output_padding_mask = output_padding_mask.unsqueeze(1).to(input_padding_mask.dtype)

    attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(output_padding_mask, -1)
    batch_size, _, output_length, input_length = attn_mask.shape
    cum_duration = torch.cumsum(duration, -1).view(batch_size * input_length, 1)
    indices = torch.arange(output_length, dtype=duration.dtype, device=duration.device)
    valid_indices = indices.unsqueeze(0) < cum_duration
    valid_indices = valid_indices.to(attn_mask.dtype).view(batch_size, input_length, output_length)
    padded_indices = valid_indices - nn.functional.pad(valid_indices, [0, 0, 1, 0, 0, 0])[:, :-1]
    attn = padded_indices.unsqueeze(1).transpose(2, 3) * attn_mask

    prior_means = torch.matmul(attn.squeeze(1), prior_means).transpose(1, 2)
    prior_log_variances = torch.matmul(attn.squeeze(1), prior_log_variances).transpose(1, 2)

    prior_latents = prior_means + torch.randn_like(prior_means) * torch.exp(prior_log_variances) * self.noise_scale
    latents = self.flow(prior_latents, output_padding_mask, speaker_embeddings, reverse=True)

    spectrogram = latents * output_padding_mask
    waveform = self.decoder(spectrogram, speaker_embeddings)
    waveform = waveform.squeeze(1)
    upsample_factor = 1
    for r in self.config.upsample_rates:
        upsample_factor *= r
    sequence_lengths = predicted_lengths * upsample_factor

    if not return_dict:
        outputs = (waveform, sequence_lengths, spectrogram) + text_encoder_output[3:]
        return outputs

    return VitsModelOutput(
        waveform=waveform,
        sequence_lengths=sequence_lengths,
        spectrogram=spectrogram,
        hidden_states=text_encoder_output.hidden_states,
        attentions=text_encoder_output.attentions,
    )


from transformers.models.vits.modeling_vits import VitsModel

VitsModel.forward = _patched_vits_forward


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
