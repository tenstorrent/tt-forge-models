# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-component nn.Module wrappers for XTTS-v2 bring-up under tt-xla.

Track 1 brings up XTTS-v2 one ``nn.Module`` at a time so each graph can be
compiled and PCC-compared against CPU in isolation, before Track 2 glues them
into an end-to-end pipeline with a host-driven autoregressive loop.

XTTS-v2 has no single traceable ``forward`` (``Xtts.forward`` is a
``NotImplementedError`` stub and ``Xtts.inference`` interleaves a stochastic,
dynamic-length ``gpt.generate`` loop with the traceable compute). Instead of
monkey-patching a fused forward, we expose each underlying ``nn.Module`` through
a thin wrapper with a clean tensor-in / tensor-out signature:

    SpeakerEncoderWrapper   mel spectrogram (CPU STFT) -> speaker embedding
    ConditioningWrapper     reference mel-spectrogram  -> GPT conditioning latents
    GptPrefillWrapper       [prefix, start] tokens     -> first-step audio logits
    GptLatentsWrapper       text + audio-code sequence -> GPT latents
    HifiganDecoderWrapper   GPT latents + speaker emb  -> 24 kHz waveform

The distinct nn.Modules are the speaker encoder, the conditioning encoder +
perceiver, the shared GPT2 transformer trunk (exercised here in two input
modes: prefill logits and full-sequence latents) and the HiFi-GAN vocoder.
"""

import torch
from torch import nn

# Restore ``isin_mps_friendly`` (dropped in transformers>=5) before TTS is
# imported anywhere -- the TTS import chain (tortoise.autoregressive) needs it.
import transformers.pytorch_utils as _pu

if not hasattr(_pu, "isin_mps_friendly"):
    _pu.isin_mps_friendly = lambda elements, test_elements: torch.isin(
        elements, test_elements
    )


class SpeakerEncoderWrapper(nn.Module):
    """ResNet-SE speaker encoder trunk: mel spectrogram ``(b, 64, T)`` -> embedding ``(b, 512)``.

    The encoder's mel front-end (``torch_spec``) runs ``torch.stft``, a complex
    FFT that lowers to an unsupported ``XLAComplexFloatType`` on device. STFT/mel
    is fixed DSP preprocessing (problem #5216 keeps pre/post-processing on CPU),
    so the loader computes the mel spectrogram on CPU and this wrapper runs only
    the learned trunk (instance norm + ResNet + attentive pooling + fc) on device.
    ``use_torch_spec`` is disabled so ``forward`` consumes the precomputed mel
    directly; feeding the mel back this way is numerically identical to the full
    waveform path (verified bit-exact on CPU).
    """

    def __init__(self, xtts):
        super().__init__()
        self.speaker_encoder = xtts.hifigan_decoder.speaker_encoder
        self.speaker_encoder.use_torch_spec = False

    def forward(self, mel_spec):
        return self.speaker_encoder(mel_spec, l2_norm=True)


class ConditioningWrapper(nn.Module):
    """GPT conditioning encoder + perceiver resampler.

    Mirrors ``GPT.get_style_emb`` for the perceiver path: a reference
    mel-spectrogram ``(b, 80, s)`` is encoded and resampled to a fixed number of
    conditioning latents ``(b, 1024, 32)``.
    """

    def __init__(self, xtts):
        super().__init__()
        self.conditioning_encoder = xtts.gpt.conditioning_encoder
        self.conditioning_perceiver = xtts.gpt.conditioning_perceiver

    def forward(self, cond_mel):
        conds = self.conditioning_encoder(cond_mel)  # (b, 1024, s)
        conds = self.conditioning_perceiver(conds.permute(0, 2, 1)).transpose(
            1, 2
        )  # (b, 1024, 32)
        return conds


class GptPrefillWrapper(nn.Module):
    """GPT2 generation trunk + audio ``lm_head`` over the ``[prefix, start]`` sequence.

    This is the first (prefill) step of ``gpt.generate`` run as a single static
    forward with ``use_cache=False``: it covers the shared GPT2 transformer trunk
    plus the audio-token ``lm_head`` (which the latent path skips). The cached
    prefix embedding is held as a buffer so it moves with ``.to(device)``; the
    incremental KV-cached decode step (same module) is a Track-2 loop graph.
    """

    def __init__(self, xtts, prefix_emb):
        super().__init__()
        self.gpt_inference = xtts.gpt.gpt_inference
        self.register_buffer("prefix_emb", prefix_emb)

    def forward(self, gpt_inputs, attention_mask):
        # GPT2InferenceModel.forward reads cached_prefix_emb; point it at the
        # buffer so the device placement matches the rest of the graph.
        self.gpt_inference.cached_prefix_emb = self.prefix_emb
        out = self.gpt_inference(
            input_ids=gpt_inputs,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.logits


class GptLatentsWrapper(nn.Module):
    """Full-sequence ``GPT.forward(return_latent=True)``: text + audio codes -> latents ``(b, L, 1024)``.

    Runs the shared GPT2 trunk + final norm over the whole code sequence (no KV
    cache). ``gpt_codes`` is supplied as a fixed input by the loader so the graph
    is deterministic and PCC-comparable (the autoregressive sampling that would
    normally produce the codes is not part of this graph).
    """

    def __init__(self, xtts):
        super().__init__()
        self.gpt = xtts.gpt

    def forward(
        self, text_tokens, text_len, gpt_codes, expected_output_len, gpt_cond_latent
    ):
        return self.gpt(
            text_tokens,
            text_len,
            gpt_codes,
            expected_output_len,
            cond_latents=gpt_cond_latent,
            return_attentions=False,
            return_latent=True,
        )


class HifiganDecoderWrapper(nn.Module):
    """HiFi-GAN vocoder: GPT latents ``(b, L, 1024)`` + speaker emb ``(b, 512, 1)`` -> waveform ``(b, 1, S)``."""

    def __init__(self, xtts):
        super().__init__()
        self.hifigan_decoder = xtts.hifigan_decoder

    def forward(self, gpt_latents, speaker_embedding):
        return self.hifigan_decoder(gpt_latents, g=speaker_embedding)
