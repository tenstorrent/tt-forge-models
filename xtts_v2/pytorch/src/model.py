# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-component nn.Module wrappers for XTTS-v2 bring-up under tt-xla.

XTTS-v2 has no single traceable ``forward``, so each learned ``nn.Module`` is
exposed through a thin tensor-in / tensor-out wrapper that can be compiled and
PCC-compared against CPU on its own:

    SpeakerEncoderWrapper   mel spectrogram (CPU STFT) -> speaker embedding
    ConditioningWrapper     reference mel-spectrogram  -> GPT conditioning latents
    GptPrefillWrapper       [prefix, start] tokens     -> first-step audio logits
    GptDecodeWrapper        one token + static KV cache-> next-step audio logits
    GptLatentsWrapper       text + audio-code sequence -> GPT latents
    HifiganDecoderWrapper   GPT latents + speaker emb  -> 24 kHz waveform

``GptCachedStep`` (the live-cache decode step the autoregressive loop reuses) is
also defined here so the e2e pipeline and the decode-loop test share one graph.
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
    """ResNet-SE speaker encoder trunk: mel ``(b, 64, T)`` -> embedding ``(b, 512)``.

    The mel front-end (``torch_spec`` / ``torch.stft``) is a complex FFT
    unsupported on device, so the loader computes the mel on CPU and this
    wrapper runs only the learned trunk (``use_torch_spec`` disabled so ``forward``
    consumes the precomputed mel).
    """

    def __init__(self, xtts):
        super().__init__()
        self.speaker_encoder = xtts.hifigan_decoder.speaker_encoder
        self.speaker_encoder.use_torch_spec = False

    def forward(self, mel_spec):
        return self.speaker_encoder(mel_spec, l2_norm=True)


class ConditioningWrapper(nn.Module):
    """GPT conditioning encoder + perceiver (``GPT.get_style_emb``): reference mel
    ``(b, 80, s)`` -> conditioning latents ``(b, 1024, 32)``."""

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
    """GPT2 trunk + audio ``lm_head``, prefill step (``use_cache=False``): ``[prefix,
    start]`` -> first-step audio logits.

    ``prefix_emb`` is held as a buffer so it moves with ``.to(device)``.
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


class GptDecodeWrapper(nn.Module):
    """One KV-cached GPT2 decode step over a static cache prefilled on CPU: embed
    one audio token, run the GPT2 trunk (``use_cache=True``), project to audio
    logits. This is the graph the autoregressive loop reuses every token.

    The forward is *pure*: the cache is passed in as ``cache_keys``/``cache_values``
    tensors and each call rebuilds a fresh ``StaticCache`` from clones, so the
    tester's repeated CPU/TT runs stay repeatable. (XTTS's GPT2 has a null ``wpe``
    -- positions come from ``mel_pos_embedding`` -- and no ``cache_position``;
    HF's ``StaticLayer`` self-manages its write index.)
    """

    def __init__(self, xtts, valid_len, max_cache_len):
        super().__init__()
        from transformers import StaticCache

        gpt = xtts.gpt
        self.gpt2 = gpt.gpt  # HF GPT2Model (wpe is null; positions are external)
        self.mel_embedding = gpt.mel_embedding
        self.mel_pos_embedding = gpt.mel_pos_embedding
        self.final_norm = gpt.final_norm
        self.mel_head = gpt.mel_head
        self._valid_len = int(valid_len)  # prefill cumulative_length
        self._max_cache_len = int(max_cache_len)
        # Skeleton built once on CPU, outside forward: no cache object is built in
        # the compiled graph (that trips device-kwarg tensor creation on TT);
        # forward only swaps in cloned tensors.
        self._cache = StaticCache(config=gpt.gpt.config, max_cache_len=max_cache_len)

    def _load_cache(self, cache_len, cache_keys, cache_values):
        """Point the StaticCache at clones of the input K/V (never mutates inputs)."""
        # cache_keys/values are stacked over layers: [n_layer, b, n_head, L, d].
        for i, layer in enumerate(self._cache.layers):
            lk = cache_keys[i].clone()
            lv = cache_values[i].clone()
            layer.keys = lk
            layer.values = lv
            # clone: StaticLayer.update() does an in-place add_ on this tensor.
            layer.cumulative_length = cache_len.clone()
            # Mark the layer initialized so update() skips lazy allocation and
            # writes straight in at cumulative_length (matches StaticLayer state).
            layer.is_initialized = True
            layer.dtype = lk.dtype
            layer.device = lk.device
            layer.max_batch_size = lk.shape[0]
            layer.num_heads = lk.shape[1]
            layer.k_head_dim = lk.shape[-1]
            layer.v_head_dim = lv.shape[-1]
        return self._cache

    def forward(
        self, audio_ids, positions, attention_mask, cache_len, cache_keys, cache_values
    ):
        cache = self._load_cache(cache_len, cache_keys, cache_values)
        emb = self.mel_embedding(audio_ids)
        emb = emb + self.mel_pos_embedding.emb(positions).unsqueeze(0)
        out = self.gpt2(
            inputs_embeds=emb,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        hidden = self.final_norm(out.last_hidden_state)
        return self.mel_head(hidden)


class GptLatentsWrapper(nn.Module):
    """Full-sequence ``GPT.forward(return_latent=True)``: text + audio codes ->
    latents ``(b, L, 1024)`` (no KV cache; ``gpt_codes`` supplied as a fixed input
    so the graph is deterministic and PCC-comparable)."""

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


class GptCachedStep(nn.Module):
    """GPT2 trunk + audio ``lm_head`` for one KV-cached decode step, driven by a
    live HF ``StaticCache`` passed in by the caller.

    Unlike ``GptDecodeWrapper`` (which rebuilds a fresh cache from K/V tensors so
    its forward is pure for the runner's single-step PCC check), this module is
    the graph the autoregressive *loop* reuses: the caller pre-allocates a
    ``StaticCache`` to a fixed ``max_cache_len`` and feeds one token per step, so
    shapes stay constant and the graph compiles once. Audio tokens are embedded
    here (``mel_embedding`` + ``mel_pos_embedding`` at absolute positions) so all
    learned ops run on TT; the prefill step prepends the conditioning+text
    ``prefix_emb``. XTTS's GPT2 has a null ``wpe`` and ``StaticLayer`` self-manages
    its write index, so the prefill and decode graphs each compile once and every
    later step is a cache hit.

    Shared by the e2e pipeline (``pipeline.XTTSPipeline``) and the decode-loop
    bring-up test (``tests/torch/models/xtts_v2/test_xtts_decode.py``).
    """

    def __init__(self, xtts):
        super().__init__()
        gpt = xtts.gpt
        self.gpt2 = gpt.gpt  # HF GPT2Model (wpe is null; positions are external)
        self.mel_embedding = gpt.mel_embedding
        self.mel_pos_embedding = gpt.mel_pos_embedding
        self.final_norm = gpt.final_norm
        self.mel_head = gpt.mel_head

    def forward(
        self, audio_ids, positions, attention_mask, past_key_values, prefix_emb=None
    ):
        emb = self.mel_embedding(audio_ids)
        emb = emb + self.mel_pos_embedding.emb(positions).unsqueeze(0)
        if prefix_emb is not None:  # prefill only
            emb = torch.cat([prefix_emb.to(emb.dtype), emb], dim=1)
        out = self.gpt2(
            inputs_embeds=emb,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        hidden = self.final_norm(out.last_hidden_state)
        return self.mel_head(hidden)
