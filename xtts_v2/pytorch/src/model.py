# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Monkey patches for XTTS-v2 under tt-xla:

1. Restore ``isin_mps_friendly`` (dropped in transformers>=5) before importing TTS.
2. Define a deterministic ``Xtts.forward`` (native is a NotImplementedError stub):
   GPT latents -> HiFiGAN decode over the ``gpt_codes`` supplied by load_inputs.
"""

import torch
import transformers.pytorch_utils as _pu

# (1) Restore isin_mps_friendly before importing TTS (its import chain needs it).
if not hasattr(_pu, "isin_mps_friendly"):
    _pu.isin_mps_friendly = lambda elements, test_elements: torch.isin(
        elements, test_elements
    )

from TTS.tts.models.xtts import Xtts


@torch.no_grad()
def forward(
    self,
    text_tokens,
    text_len,
    gpt_codes,
    expected_output_len,
    gpt_cond_latent,
    speaker_embedding,
):
    """Deterministic XTTS-v2 compute graph (GPT latents -> HiFiGAN decode).

    Mirrors the deterministic tail of ``Xtts.inference``. ``gpt_codes`` is a
    fixed input built by ``ModelLoader.load_inputs`` (the autoregressive
    ``gpt.generate`` step is not run), keeping this graph reproducible for PCC.

    Returns the generated waveform tensor (shape ``[1, 1, S]``, 24kHz).
    """
    gpt_latents = self.gpt(
        text_tokens,
        text_len,
        gpt_codes,
        expected_output_len,
        cond_latents=gpt_cond_latent,
        return_attentions=False,
        return_latent=True,
    )
    wav = self.hifigan_decoder(gpt_latents, g=speaker_embedding)
    self._cached_audio = wav
    return wav


def post_process(self):
    """Return the waveform produced by the most recent ``forward`` call."""
    if getattr(self, "_cached_audio", None) is None:
        raise RuntimeError("post_process() called before forward().")
    return self._cached_audio


Xtts.forward = forward
Xtts.post_process = post_process
