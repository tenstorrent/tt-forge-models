# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helpers for loading Coqui XTTS-v2 and exposing its GPT-2 autoregressive
backbone (the LM that predicts the text + audio code tokens) as a single
forward-pass ``nn.Module`` suitable for Tenstorrent bringup.

XTTS-v2 is a multi-component text-to-speech pipeline (GPT backbone + HiFi-GAN
decoder + speaker/perceiver conditioning encoder + DVAE). The compute-dominant
and most device-portable part is the 30-layer GPT-2 transformer. This module
brings up that backbone: given the text tokens, audio code tokens and the
precomputed conditioning latents, it runs the concatenated
``[cond | text | mel]`` embedding sequence through the GPT-2 stack and the
text/mel prediction heads (exactly what ``GPT.get_logits`` does internally).

The speaker/perceiver conditioning encoder that produces ``cond_latents`` from
reference audio is host-side preprocessing and is not part of this backbone
graph, so the latents are supplied as an input tensor.
"""
import os

import torch
import torch.nn as nn

# XTTS-v2 fixed architecture constants (from the HF config.json model_args).
NUM_TEXT_TOKENS = 6681  # gpt_number_text_tokens -> text_embedding / text_head size
NUM_AUDIO_TOKENS = 1026  # gpt_num_audio_tokens -> mel_embedding / mel_head size
MODEL_DIM = 1024  # gpt_n_model_channels
NUM_COND_LATENTS = 32  # perceiver resampler latents

HF_REPO = "coqui/XTTS-v2"
_CHECKPOINT_FILES = ("config.json", "model.pth", "vocab.json", "speakers_xtts.pth", "mel_stats.pth")


def _install_transformers_compat_shim():
    """coqui-tts 0.27.5 imports ``isin_mps_friendly`` from
    ``transformers.pytorch_utils``, which was removed in transformers>=5. Inject
    an equivalent (plain ``torch.isin``) so the XTTS modules import cleanly.
    """
    import transformers.pytorch_utils as ptu

    if not hasattr(ptu, "isin_mps_friendly"):
        ptu.isin_mps_friendly = lambda elements, test_elements: torch.isin(
            elements, test_elements
        )


def load_xtts_model(checkpoint_dir=None):
    """Download (if needed) and load the full XTTS-v2 model.

    Returns:
        (Xtts, XttsConfig): the loaded, eval-mode model and its config.
    """
    _install_transformers_compat_shim()
    # XTTS-v2 weights are gated behind the Coqui Public Model License ToS prompt.
    os.environ.setdefault("COQUI_TOS_AGREED", "1")

    from huggingface_hub import hf_hub_download
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    cfg_path = hf_hub_download(HF_REPO, "config.json")
    for f in _CHECKPOINT_FILES:
        hf_hub_download(HF_REPO, f)
    ckpt_dir = checkpoint_dir or os.path.dirname(cfg_path)

    config = XttsConfig()
    config.load_json(cfg_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config, checkpoint_dir=ckpt_dir, eval=True, use_deepspeed=False
    )
    model.eval()
    return model, config


class _NullPositionEmbeddings(nn.Module):
    """Dtype-correct replacement for XTTS's ``null_position_embeddings``.

    XTTS zeroes out GPT-2's built-in positional embedding (it adds its own
    learned position embeddings before the transformer). The stock helper
    returns fp32 zeros, which upcasts the hidden states and breaks the bf16
    device path (GPT-2 adds the position embedding without a dtype cast). This
    version emits zeros in the model's current dtype, tracked via a real
    parameter so it follows ``.to(dtype)``.
    """

    def __init__(self, dim, dtype_ref):
        super().__init__()
        self.dim = dim
        self._dtype_ref = dtype_ref  # an nn.Parameter that follows .to(dtype)

    def forward(self, position_ids):
        return torch.zeros(
            position_ids.shape[0],
            position_ids.shape[1],
            self.dim,
            dtype=self._dtype_ref.dtype,
            device=position_ids.device,
        )


class XttsGptBackbone(nn.Module):
    """Single forward pass over the XTTS-v2 GPT-2 backbone.

    Computes text/mel input embeddings, prepends the conditioning latents, runs
    the GPT-2 transformer stack and applies the text and mel prediction heads,
    returning the per-position logits over the text and audio-code vocabularies.
    """

    def __init__(self, gpt):
        super().__init__()
        # ``gpt`` is the TTS ``GPT`` module (holds the GPT2Model transformer in
        # ``gpt.gpt`` plus the embeddings, position embeddings and heads).
        self.gpt = gpt
        # Swap the fp32 null positional embedding for a dtype-tracking one so the
        # bf16 device path stays consistent (see _NullPositionEmbeddings).
        dtype_ref = gpt.gpt.h[0].ln_1.weight
        gpt.gpt.wpe = _NullPositionEmbeddings(gpt.model_dim, dtype_ref)

    def forward(self, text_inputs, audio_codes, cond_latents):
        gpt = self.gpt
        text_emb = gpt.text_embedding(text_inputs) + gpt.text_pos_embedding(text_inputs)
        mel_emb = gpt.mel_embedding(audio_codes) + gpt.mel_pos_embedding(audio_codes)
        text_logits, mel_logits = gpt.get_logits(
            text_emb,
            gpt.text_head,
            mel_emb,
            gpt.mel_head,
            prompt=cond_latents,
        )
        return {"text_logits": text_logits, "mel_logits": mel_logits}
