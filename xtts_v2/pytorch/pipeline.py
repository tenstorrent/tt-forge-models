# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""XTTS-v2 (coqui/XTTS-v2) — end-to-end text-to-speech pipeline on Tenstorrent.

Chains the per-component nn.Module graphs (loader + wrappers in this package)
into the full ``Xtts.inference`` path, running the learned modules on TT and
keeping fixed DSP / orchestration on CPU. Reusable implementation shared by the
tt-xla e2e test; all building blocks and inputs come from this package's loader.

Stages (matching ``Xtts.inference`` + ``Xtts.get_conditioning_latents``):

    reference wav ─(CPU mel)──────► speaker_encoder  [TT] ─► speaker_embedding
    reference wav ─(CPU mel)──────► conditioning     [TT] ─► gpt_cond_latent
    text ─(CPU tokenizer)─► gpt autoregressive loop  [TT] ─► gpt_codes
    text + gpt_codes ─────────────► gpt_latents      [TT] ─► gpt_latents
    gpt_latents + speaker_embedding► hifigan_decoder  [TT] ─► 24 kHz waveform

CPU handles only non-learned work (mel front-ends, tokenizer, sampling / loop
control, audio I/O); sampling/conditioning use the reference ``Xtts.inference``
params so behavior matches the source. The audio-token loop drives
``GptCachedStep`` (GPT2 trunk on TT) over a pre-allocated HF ``StaticCache``:
fixed-shape inputs mean the decode graph compiles once and is reused every step.
``max_audio_tokens`` bounds ``max_cache_len``.

Requires the optional ``coqui-tts`` + ``torchaudio`` deps and the CPML-gated
coqui/XTTS-v2 weights (``COQUI_TOS_AGREED=1``).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from loguru import logger

from .loader import ModelLoader, ModelVariant
from .src.model import (
    ConditioningWrapper,
    GptCachedStep,
    GptLatentsWrapper,
    HifiganDecoderWrapper,
    SpeakerEncoderWrapper,
)

# Model-card example text; any real sentence works.
DEFAULT_TEXT = (
    "It took me quite a long time to develop a voice, and now that I have it "
    "I'm not going to be silent."
)
DEFAULT_LANGUAGE = "en"
OUTPUT_SAMPLE_RATE = 24000

# Reference autoregressive sampling params, matching ``Xtts.inference`` defaults
# (coqui/XTTS-v2). The decode loop applies these host-side via HF's own logits
# processors so token selection matches the reference generate() exactly.
REF_TEMPERATURE = 0.75
REF_TOP_K = 50
REF_TOP_P = 0.85
REF_REPETITION_PENALTY = 10.0
# Reference conditioning params, matching ``Xtts.get_conditioning_latents``
# defaults: gpt_cond_len == gpt_cond_chunk_len == 6 -> a single 6 s chunk, with
# multi-chunk mean when they differ (see ``Xtts.get_gpt_cond_latents``).
GPT_COND_LEN = 6
GPT_COND_CHUNK_LEN = 6
MIN_AUDIO_SECONDS = 0.33  # chunks shorter than this are skipped (reference)


class XTTSConfig:
    def __init__(
        self,
        text: str = DEFAULT_TEXT,
        language: str = DEFAULT_LANGUAGE,
        speaker_wav: Optional[str] = None,
        max_audio_tokens: Optional[int] = None,
        seed: int = 0,
    ):
        self.text = text
        self.language = language
        # A reference speaker clip; defaults to the same public LibriSpeech
        # utterance the component loader uses so the pipeline runs out of the box.
        self.speaker_wav = speaker_wav
        # Cap on generated audio tokens (each ~= 1024 output samples / 24 kHz);
        # keeps the single-compile TT decode loop demo-sized. None = model max.
        self.max_audio_tokens = max_audio_tokens
        # Seed for the (stochastic) reference sampling, so runs are reproducible.
        self.seed = seed


class XTTSPipeline:
    """coqui/XTTS-v2 text-to-speech pipeline (components chained on TT)."""

    def __init__(self, config: XTTSConfig):
        self.config = config
        self._loader = ModelLoader(variant=ModelVariant.GPT_LATENTS)

    # ------------------------------------------------------------------ #
    # setup: build the full model once, wrap each component, compile for TT
    # ------------------------------------------------------------------ #
    def setup(self):
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        self.xtts = self._loader._build_xtts()  # downloads weights, eval mode
        gpt = self.xtts.gpt

        self.start_audio_token = gpt.start_audio_token
        self.stop_audio_token = gpt.stop_audio_token
        self.code_stride_len = gpt.code_stride_len
        self.model_max_audio_tokens = gpt.max_gen_mel_tokens

        # Component graphs (same wrappers as the bring-up tests). Register the
        # `tt` backend now; the actual .to(device) happens per stage in run().
        self.speaker_encoder = SpeakerEncoderWrapper(self.xtts).eval()
        self.conditioning = ConditioningWrapper(self.xtts).eval()
        self.gpt_latents = GptLatentsWrapper(self.xtts).eval()
        self.hifigan = HifiganDecoderWrapper(self.xtts).eval()

        self.speaker_encoder.compile(backend="tt")
        self.conditioning.compile(backend="tt")
        self.gpt_latents.compile(backend="tt")
        self.hifigan.compile(backend="tt")

        self.decode_step = GptCachedStep(self.xtts).eval()
        self.decode_step.compile(backend="tt")

    # ------------------------------------------------------------------ #
    # CPU preprocessing (mirrors get_conditioning_latents + tokenizer)
    # ------------------------------------------------------------------ #
    def _reference_audio_22k(self) -> torch.Tensor:
        import numpy as np
        import torchaudio

        if self.config.speaker_wav:
            audio, sr = torchaudio.load(self.config.speaker_wav)
            audio = audio.mean(0, keepdim=True)  # mono
        else:
            # Same public LibriSpeech reference the component loader uses.
            from ...tools.utils import get_file

            sample = torch.load(
                get_file(self._loader.REFERENCE_AUDIO), weights_only=False
            )
            sr = int(sample["audio"].get("sampling_rate", 16000))
            audio = torch.tensor(
                np.asarray(sample["audio"]["array"], dtype="float32")
            ).unsqueeze(0)
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        return audio

    def _speaker_mel(self, audio22: torch.Tensor) -> torch.Tensor:
        """16 kHz reference -> speaker-encoder mel (computed on CPU)."""
        import torchaudio

        audio16 = torchaudio.functional.resample(audio22, 22050, 16000)
        # use_torch_spec is disabled inside the wrapper, so feed the mel directly.
        with torch.no_grad():
            return self.xtts.hifigan_decoder.speaker_encoder.torch_spec(audio16)

    def _conditioning_mels(self, audio22: torch.Tensor) -> list:
        """Reference ``get_gpt_cond_latents`` chunking: split the first
        ``GPT_COND_LEN`` s into ``GPT_COND_CHUNK_LEN``-second mel chunks; the
        conditioning encoder runs per chunk on TT and ``run()`` means over chunks.
        Reference defaults (6 == 6) give a single 6 s chunk."""
        from TTS.tts.models.xtts import wav_to_mel_cloning

        audio = audio22[:, : 22050 * GPT_COND_LEN]
        step = 22050 * GPT_COND_CHUNK_LEN
        mels = []
        for i in range(0, audio.shape[1], step):
            chunk = audio[:, i : i + step]
            if chunk.size(-1) < 22050 * MIN_AUDIO_SECONDS:
                continue  # skip too-short trailing chunk (reference behavior)
            mels.append(
                wav_to_mel_cloning(
                    chunk,
                    mel_norms=self.xtts.mel_stats.cpu(),
                    n_fft=2048,
                    hop_length=256,
                    win_length=1024,
                    power=2,
                    normalized=False,
                    sample_rate=22050,
                    f_min=0,
                    f_max=8000,
                    n_mels=80,
                )
            )
        if not mels:
            raise RuntimeError(
                f"Reference audio too short (< {MIN_AUDIO_SECONDS:.2f}s) for conditioning."
            )
        return mels

    def _text_tokens(self) -> torch.Tensor:
        toks = self.xtts.tokenizer.encode(
            self.config.text.strip().lower(), lang=self.config.language
        )
        return torch.IntTensor(toks).unsqueeze(0)

    # ------------------------------------------------------------------ #
    # gpt_codes: autoregressive audio-token loop
    # ------------------------------------------------------------------ #
    def _make_static_cache(self, max_cache_len: int, device):
        """StaticCache for XTTS's GPT2, built on CPU then moved to device
        (build + early_initialization on CPU avoids a trace/fusion issue,
        tt-xla#1645). GPT2's StaticLayer self-manages its write index."""
        from transformers import StaticCache

        cfg = self.xtts.gpt.gpt.config
        n_head = cfg.num_attention_heads
        head_dim = cfg.hidden_size // n_head
        cache = StaticCache(
            config=cfg,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device="cpu",
            dtype=torch.float32,
        )
        cache.early_initialization(
            batch_size=1,
            num_heads=n_head,
            head_dim=head_dim,
            dtype=torch.float32,
            device="cpu",
        )
        if device != "cpu":
            for layer in cache.layers:
                layer.keys = layer.keys.to(device)
                layer.values = layer.values.to(device)
                layer.cumulative_length = layer.cumulative_length.to(device)
                layer.device = device
        return cache

    def _generate_codes_tt(self, gpt_cond_latent, text_tokens) -> torch.Tensor:
        """KV-cached, single-compile decode loop with the GPT2 trunk on TT.

        Prefills the conditioning+text prefix + [START] into a StaticCache, then
        samples audio tokens one at a time with the reference ``Xtts.inference``
        params (host-side HF logits processors, so selection matches
        ``gpt.generate``). The TT decode graph feeds only the new token with
        constant shapes, so it compiles once and is reused every step.
        """
        from transformers import (
            LogitsProcessorList,
            RepetitionPenaltyLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )

        device = xm.xla_device()
        gpt = self.xtts.gpt

        # Prefix embedding (cond latents + [START]text[STOP]); host side.
        with torch.no_grad():
            gpt.compute_embeddings(gpt_cond_latent, text_tokens)
        prefix_emb = gpt.gpt_inference.cached_prefix_emb.clone().to(
            device
        )  # [1,P,1024]
        prefix_len = prefix_emb.shape[1]

        max_tokens = self.config.max_audio_tokens or self.model_max_audio_tokens
        max_tokens = int(min(max_tokens, self.model_max_audio_tokens))
        max_cache_len = prefix_len + max_tokens

        cache = self._make_static_cache(max_cache_len, device)
        self.decode_step = self.decode_step.to(device)

        def mask(valid):  # [1, max_cache_len]; 1s for written cache slots
            m = torch.zeros((1, max_cache_len), dtype=torch.long)
            m[:, :valid] = 1
            return m.to(device)

        # Reference sampling stack: repetition penalty (processor) then the
        # temperature/top-k/top-p warpers, in HF's generate() order. Applied on
        # the CPU logits each step; the running audio-token sequence (starting
        # with [START]) drives the repetition penalty, exactly like HF generate.
        processors = LogitsProcessorList(
            [
                RepetitionPenaltyLogitsProcessor(penalty=REF_REPETITION_PENALTY),
                TemperatureLogitsWarper(REF_TEMPERATURE),
                TopKLogitsWarper(REF_TOP_K),
                TopPLogitsWarper(REF_TOP_P),
            ]
        )
        rng = torch.Generator().manual_seed(int(self.config.seed))
        seq = [self.start_audio_token]  # running sequence for repetition penalty

        def sample(logits_row_cpu):  # logits_row_cpu: [1, vocab] on CPU
            input_ids = torch.tensor([seq], dtype=torch.long)
            scores = processors(input_ids, logits_row_cpu.float())
            probs = torch.softmax(scores, dim=-1)
            return int(torch.multinomial(probs, num_samples=1, generator=rng).item())

        generated = []
        with torch.no_grad():
            # --- Prefill: [prefix, START(audio pos 0)] -> first audio token ---
            start_ids = torch.tensor(
                [[self.start_audio_token]], dtype=torch.long, device=device
            )
            pos0 = torch.tensor([0], dtype=torch.long, device=device)
            logits = self.decode_step(
                start_ids, pos0, mask(prefix_len + 1), cache, prefix_emb
            )
            next_token = sample(logits[:, -1, :].to("cpu"))

            cur = prefix_len + 1  # cache positions written so far
            # --- Decode loop: feed 1 token per step, audio position = step ---
            for step in range(1, max_tokens):
                if next_token == self.stop_audio_token:
                    break
                generated.append(next_token)
                seq.append(next_token)  # extend history before sampling the next
                tok = torch.tensor([[next_token]], dtype=torch.long, device=device)
                pos = torch.tensor([step], dtype=torch.long, device=device)
                logits = self.decode_step(tok, pos, mask(cur + 1), cache, None)
                next_token = sample(logits[:, -1, :].to("cpu"))
                cur += 1
                if step % 32 == 0:
                    logger.info(f"[gpt_codes] {step}/{max_tokens} tokens")

        self.decode_step = self.decode_step.to("cpu")
        return torch.tensor(generated, dtype=torch.long).unsqueeze(0)

    # ------------------------------------------------------------------ #
    # run: full text -> waveform
    # ------------------------------------------------------------------ #
    def run(self) -> torch.Tensor:
        device = xm.xla_device()
        tt = lambda x: x.to(device)
        cpu = lambda x: x.to("cpu")

        with torch.no_grad():
            # --- CPU preprocessing ---
            audio22 = self._reference_audio_22k()
            speaker_mel = self._speaker_mel(audio22)
            cond_mels = self._conditioning_mels(audio22)
            text_tokens = self._text_tokens()
            text_len = torch.tensor([text_tokens.shape[-1]])

            # --- speaker_embedding [TT] ---
            logger.info("[STAGE] speaker_encoder (TT)")
            self.speaker_encoder = self.speaker_encoder.to(device)
            speaker_embedding = cpu(self.speaker_encoder(tt(speaker_mel)))
            self.speaker_encoder = self.speaker_encoder.to("cpu")
            # Match get_speaker_embedding output shape [1, 512, 1].
            speaker_embedding = speaker_embedding.unsqueeze(-1)

            # --- gpt_cond_latent [TT] ---
            # Per-chunk conditioning on TT, then mean over chunks (reference
            # get_gpt_cond_latents). Each chunk's style_emb is [1, 1024, 32].
            logger.info("[STAGE] conditioning encoder (TT)")
            self.conditioning = self.conditioning.to(device)
            style_embs = [cpu(self.conditioning(tt(m))) for m in cond_mels]
            self.conditioning = self.conditioning.to("cpu")
            conds = torch.stack(style_embs).mean(dim=0)  # [1, 1024, 32]
            gpt_cond_latent = conds.transpose(1, 2)  # [1, 32, 1024]

            # --- gpt_codes (autoregressive, GPT2 trunk on TT) ---
            logger.info("[STAGE] gpt_codes (TT)")
            gpt_codes = self._generate_codes_tt(gpt_cond_latent, text_tokens)
            logger.info(f"[gpt_codes] produced {gpt_codes.shape[-1]} audio tokens")

            expected_output_len = torch.tensor(
                [gpt_codes.shape[-1] * self.code_stride_len]
            )

            # --- gpt_latents [TT] ---
            logger.info("[STAGE] gpt_latents (TT)")
            self.gpt_latents = self.gpt_latents.to(device)
            gpt_latents = cpu(
                self.gpt_latents(
                    tt(text_tokens),
                    tt(text_len),
                    tt(gpt_codes),
                    tt(expected_output_len),
                    tt(gpt_cond_latent),
                )
            )
            self.gpt_latents = self.gpt_latents.to("cpu")

            # --- waveform [TT] ---
            logger.info("[STAGE] hifigan_decoder (TT)")
            self.hifigan = self.hifigan.to(device)
            wav = cpu(self.hifigan(tt(gpt_latents), tt(speaker_embedding)))
            self.hifigan = self.hifigan.to("cpu")

        return wav  # [1, 1, S] @ 24 kHz


def save_wav(wav: torch.Tensor, filepath: str = "xtts_output.wav"):
    """Save the pipeline waveform as a 24 kHz 16-bit PCM WAV.

    Uses the stdlib ``wave`` module rather than ``torchaudio.save`` so it does
    not depend on FFmpeg / torchcodec (often absent on bring-up hosts).
    """
    import wave

    audio = wav.detach().cpu().float().reshape(-1)  # mono [S]
    audio = torch.clamp(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).round().to(torch.int16).numpy()
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(OUTPUT_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return filepath


def run_xtts_pipeline(
    output_path: str = "xtts_output.wav",
    text: str = DEFAULT_TEXT,
    language: str = DEFAULT_LANGUAGE,
    speaker_wav: Optional[str] = None,
    max_audio_tokens: Optional[int] = None,
    seed: int = 0,
) -> torch.Tensor:
    """Run the XTTS-v2 pipeline and write a WAV file. Returns the waveform tensor."""
    # optimization_level 0: the memory-layout optimizer probes ttnn op
    # constraints by allocating buffers on-device at compile time, which can
    # OOM when an earlier stage's weights are still resident. Level 0 skips it.
    torch_xla.set_custom_compile_options({"optimization_level": 0})

    config = XTTSConfig(
        text=text,
        language=language,
        speaker_wav=speaker_wav,
        max_audio_tokens=max_audio_tokens,
        seed=seed,
    )
    pipeline = XTTSPipeline(config)
    pipeline.setup()
    wav = pipeline.run()
    save_wav(wav, output_path)
    logger.info(
        f"[XTTS] wrote {output_path} ({wav.shape[-1]} samples @ {OUTPUT_SAMPLE_RATE} Hz)"
    )
    return wav


if __name__ == "__main__":
    import argparse

    import torch_xla.runtime as xr

    parser = argparse.ArgumentParser(description="XTTS-v2 e2e text-to-speech on TT")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument(
        "--speaker-wav",
        default=None,
        help="Reference speaker WAV (defaults to a public LibriSpeech clip)",
    )
    parser.add_argument("--output", default="xtts_output.wav")
    parser.add_argument(
        "--max-audio-tokens",
        type=int,
        default=None,
        help="Cap on generated audio tokens (keeps the TT decode demo short)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the reference stochastic sampling (reproducible runs)",
    )
    args = parser.parse_args()

    # torch_xla defaults to CPU; point it at the Tenstorrent device.
    xr.set_device_type("TT")

    run_xtts_pipeline(
        output_path=args.output,
        text=args.text,
        language=args.language,
        speaker_wav=args.speaker_wav,
        max_audio_tokens=args.max_audio_tokens,
        seed=args.seed,
    )
