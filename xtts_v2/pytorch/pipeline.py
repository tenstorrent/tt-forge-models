# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""XTTS-v2 (coqui/XTTS-v2) end-to-end text-to-speech pipeline on Tenstorrent.

Chains the per-component nn.Module graphs into the full ``Xtts.inference`` path,
running the learned modules on TT and keeping fixed DSP/orchestration on CPU. The
audio-token loop reuses one ``GptCachedStep`` graph over a StaticCache (compiles
once). Requires ``coqui-tts`` + ``torchaudio`` and CPML-gated weights (``COQUI_TOS_AGREED=1``).
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Optional

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from loguru import logger

from .loader import ModelLoader, ModelVariant
from .src.model import (
    ConditioningWrapper,
    GptCachedStep,
    GptLatentsWrapper,
    HifiganDecoderWrapper,
    SpeakerEncoderWrapper,
)


@contextmanager
def _timed(store: list, label: str):
    """Wall-clock time the enclosed block and append ``(label, seconds)`` to
    ``store``. Used to build the end-of-run stage-timing table. TT stages are
    materialized (``.to("cpu")``) inside the block, so the measured time
    includes the one-time per-stage TT compilation on its first (only) call."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store.append((label, time.perf_counter() - t0))


def _print_stats_table(setup_secs: float, per_run: list, warmup: int):
    """Print per-stage min/max/avg over the measured runs. ``per_run`` is a list
    (one entry per measured run) of ``{stage_label: seconds}`` dicts. ``setup``
    (model build + compile registration) is a one-time cost, reported separately.
    Uses ``print`` (stdout) so it is captured reliably under output redirection."""
    if not per_run:
        return
    n = len(per_run)
    stages = list(per_run[0].keys())  # insertion order == pipeline order

    def agg(vals):
        return min(vals), max(vals), sum(vals) / len(vals)

    label_w = max([len("Stage"), len("TOTAL (per run)")] + [len(s) for s in stages])
    bar = "=" * (label_w + 33)
    rows = [
        "",
        bar,
        f"XTTS-v2 pipeline timing over {n} run(s), after {warmup} warmup (seconds)",
        bar,
        f"setup: model build + compile registration: {setup_secs:.3f}  (one-time, excluded below)",
        "-" * (label_w + 33),
        f"{'Stage':<{label_w}}  {'min':>9}  {'max':>9}  {'avg':>9}",
        "-" * (label_w + 33),
    ]
    for s in stages:
        mn, mx, av = agg([r[s] for r in per_run])
        rows.append(f"{s:<{label_w}}  {mn:>9.3f}  {mx:>9.3f}  {av:>9.3f}")
    mn, mx, av = agg([sum(r.values()) for r in per_run])
    rows.append("-" * (label_w + 33))
    rows.append(f"{'TOTAL (per run)':<{label_w}}  {mn:>9.3f}  {mx:>9.3f}  {av:>9.3f}")
    rows.append(bar)
    rows.append("(measured runs are warm: compiled + on-device kernels cached during warmup)")
    print("\n".join(rows), flush=True)

# Model-card example text; any real sentence works.
DEFAULT_TEXT = (
    "It took me quite a long time to develop a voice, and now that I have it "
    "I'm not going to be silent."
)
DEFAULT_LANGUAGE = "en"
OUTPUT_SAMPLE_RATE = 24000

# Persistent PJRT compilation cache. Compiled TT executables are keyed by the
# StableHLO graph, so a "warm" run reuses them and skips recompilation. Kept
# under $HOME (stable) rather than the repo tree so it survives across runs.
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/tt-xla/xtts_pjrt_cache")

# Audio sample rates (Hz).
XTTS_SAMPLE_RATE = 22050  # XTTS native rate for conditioning / cloning mels
SPEAKER_ENCODER_SR = 16000  # speaker-encoder input rate
DEFAULT_REFERENCE_SR = 16000  # fallback rate of the packaged reference clip

# wav_to_mel_cloning front-end params (reference Xtts.get_gpt_cond_latents).
MEL_N_FFT = 2048
MEL_HOP_LENGTH = 256
MEL_WIN_LENGTH = 1024
MEL_POWER = 2
MEL_F_MIN = 0
MEL_F_MAX = 8000
MEL_N_MELS = 80

# Emit a progress log every N decode tokens.
DECODE_LOG_INTERVAL = 32

# Bring-up workaround for a tt-mlir tile-alignment limitation. The conditioning
# encoder's AttentionBlocks apply a ``ttnn.group_norm`` whose flattened height
# must be a multiple of 32 (tile-aligned), otherwise TTIRToTTNNCommon lowering
# fails ("flattened height must be tile-aligned, got 505"). That flattened
# height is the mel time-frame count T: ConditioningEncoder.init is a kernel-1
# Conv1d and every conv inside AttentionBlock is kernel-1 stride-1, so there is
# NO time downsampling before the group_norm (downsampling factor D = 1). Hence
# we pad T up to the next multiple of 32. The downstream PerceiverResampler
# collapses the (variable-length) time axis to a fixed 32 latents, so padding
# does not change the [1, 1024, 32] conditioning-latent output shape.
COND_MEL_TIME_ALIGN = 32

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
        self.timings = []  # list[(stage_label, seconds)] for the end-of-run table
        self.tp_mesh = None  # set for multi-chip TP (see run_xtts_pipeline / sharding.py)
        self._tp_applied = False

    def _apply_tp_sharding_once(self):
        """When running multi-chip (``tp_mesh`` set), mark the GPT2 trunk weights
        with the Megatron TP spec. Idempotent: applied on the first GPT stage
        (weights are on-device by then) and skipped thereafter. No-op for tp=1."""
        if self.tp_mesh is None or self._tp_applied:
            return
        from .sharding import apply_gpt_tp_sharding

        n = apply_gpt_tp_sharding(self.xtts.gpt, self.tp_mesh)
        self._tp_applied = True
        logger.info(f"[XTTS][TP] sharded {n} GPT tensors onto mesh {self.tp_mesh}")

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
            sr = int(sample["audio"].get("sampling_rate", DEFAULT_REFERENCE_SR))
            audio = torch.tensor(
                np.asarray(sample["audio"]["array"], dtype="float32")
            ).unsqueeze(0)
        if sr != XTTS_SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, XTTS_SAMPLE_RATE)
        return audio

    def _speaker_mel(self, audio22: torch.Tensor) -> torch.Tensor:
        """16 kHz reference -> speaker-encoder mel (computed on CPU)."""
        import torchaudio

        audio16 = torchaudio.functional.resample(
            audio22, XTTS_SAMPLE_RATE, SPEAKER_ENCODER_SR
        )
        # use_torch_spec is disabled inside the wrapper, so feed the mel directly.
        with torch.no_grad():
            return self.xtts.hifigan_decoder.speaker_encoder.torch_spec(audio16)

    @staticmethod
    def _pad_mel_time(mel: torch.Tensor, multiple: int) -> torch.Tensor:
        """Pad a ``[1, 80, T]`` mel's time axis (last dim) up to the next multiple
        of ``multiple``, on the host, so the conditioning encoder's group_norm
        flattened height is tile-aligned (see COND_MEL_TIME_ALIGN). Uses
        edge-replication (repeat the last frame) rather than zero-padding to
        minimize disturbance to the group_norm statistics. Bring-up workaround
        for a tt-mlir tile-alignment compiler limitation."""
        t = mel.size(-1)
        pad = (-t) % multiple
        if pad == 0:
            return mel
        # replicate the last time frame `pad` times: F.pad with mode="replicate"
        # expects a 3-D (N, C, W) input, which our [1, 80, T] mel already is.
        return torch.nn.functional.pad(mel, (0, pad), mode="replicate")

    def _conditioning_mels(self, audio22: torch.Tensor) -> list:
        """Reference ``get_gpt_cond_latents`` chunking into GPT_COND_CHUNK_LEN-second
        mel chunks; the conditioning encoder runs per chunk on TT, meaned in run()."""
        from TTS.tts.models.xtts import wav_to_mel_cloning

        audio = audio22[:, : XTTS_SAMPLE_RATE * GPT_COND_LEN]
        step = XTTS_SAMPLE_RATE * GPT_COND_CHUNK_LEN
        mels = []
        for i in range(0, audio.shape[1], step):
            chunk = audio[:, i : i + step]
            if chunk.size(-1) < XTTS_SAMPLE_RATE * MIN_AUDIO_SECONDS:
                continue  # skip too-short trailing chunk (reference behavior)
            mel = wav_to_mel_cloning(
                chunk,
                mel_norms=self.xtts.mel_stats.cpu(),
                n_fft=MEL_N_FFT,
                hop_length=MEL_HOP_LENGTH,
                win_length=MEL_WIN_LENGTH,
                power=MEL_POWER,
                normalized=False,
                sample_rate=XTTS_SAMPLE_RATE,
                f_min=MEL_F_MIN,
                f_max=MEL_F_MAX,
                n_mels=MEL_N_MELS,
            )
            # Host-side tile-alignment pad of the mel time axis before the mel is
            # sent to the TT conditioning encoder (see COND_MEL_TIME_ALIGN).
            mel = self._pad_mel_time(mel, COND_MEL_TIME_ALIGN)
            mels.append(mel)
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
        params (host-side HF logits processors). Constant-shape inputs mean the
        decode graph compiles once and is reused every step.
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
        self._apply_tp_sharding_once()  # multi-chip: shard GPT weights (no-op for tp=1)

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
                if step % DECODE_LOG_INTERVAL == 0:
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

        self.timings = []  # fresh per-call stage timings (see run_xtts_pipeline)
        with torch.no_grad():
            # --- CPU preprocessing ---
            with _timed(self.timings, "CPU preprocessing (mel + tokenize)"):
                audio22 = self._reference_audio_22k()
                speaker_mel = self._speaker_mel(audio22)
                cond_mels = self._conditioning_mels(audio22)
                text_tokens = self._text_tokens()
                text_len = torch.tensor([text_tokens.shape[-1]])

            # --- speaker_embedding [TT] ---
            logger.info("[STAGE] speaker_encoder (TT)")
            with _timed(self.timings, "speaker_encoder (TT)"):
                self.speaker_encoder = self.speaker_encoder.to(device)
                speaker_embedding = cpu(self.speaker_encoder(tt(speaker_mel)))
                self.speaker_encoder = self.speaker_encoder.to("cpu")
            # Match get_speaker_embedding output shape [1, 512, 1].
            speaker_embedding = speaker_embedding.unsqueeze(-1)

            # --- gpt_cond_latent [TT] ---
            # Per-chunk conditioning on TT, then mean over chunks (reference
            # get_gpt_cond_latents). Each chunk's style_emb is [1, 1024, 32].
            logger.info("[STAGE] conditioning encoder (TT)")
            with _timed(self.timings, "conditioning encoder (TT)"):
                self.conditioning = self.conditioning.to(device)
                style_embs = [cpu(self.conditioning(tt(m))) for m in cond_mels]
                self.conditioning = self.conditioning.to("cpu")
                conds = torch.stack(style_embs).mean(dim=0)  # [1, 1024, 32]
                gpt_cond_latent = conds.transpose(1, 2)  # [1, 32, 1024]

            # --- gpt_codes (autoregressive, GPT2 trunk on TT) ---
            logger.info("[STAGE] gpt_codes (TT)")
            with _timed(self.timings, "gpt_codes decode loop (TT)"):
                gpt_codes = self._generate_codes_tt(gpt_cond_latent, text_tokens)
            logger.info(f"[gpt_codes] produced {gpt_codes.shape[-1]} audio tokens")

            expected_output_len = torch.tensor(
                [gpt_codes.shape[-1] * self.code_stride_len]
            )

            # --- gpt_latents [TT] ---
            logger.info("[STAGE] gpt_latents (TT)")
            with _timed(self.timings, "gpt_latents (TT)"):
                self.gpt_latents = self.gpt_latents.to(device)
                self._apply_tp_sharding_once()  # covers decode-loop-skipped path
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
            with _timed(self.timings, "hifigan_decoder (TT)"):
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
    warmup: int = 1,
    repeat: int = 3,
    opt_level: int = 0,
    tp: int = 1,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
) -> torch.Tensor:
    """Run the XTTS-v2 pipeline and write a WAV file. Returns the waveform tensor.

    Builds the model once, runs ``warmup`` un-timed iterations (to pay the
    one-time compile + on-device kernel-caching cost), then ``repeat`` measured
    iterations and prints a per-stage min/max/avg table. The WAV from the last
    iteration is saved.

    ``tp`` > 1 enables multi-chip tensor parallelism on the GPT (Megatron TP; see
    sharding.py + SHARDING_TP2.md). tp=1 is the unchanged single-chip path. tp>1
    requires a machine with ``device_count % tp == 0`` (e.g. tp=2 on an n300).

    ``cache_dir`` enables the persistent PJRT compilation cache so subsequent
    runs reuse compiled TT executables (no recompilation). Pass ``None``/"" to
    disable. The device type must already be set (``xr.set_device_type("TT")``).
    """
    # Multi-chip tensor parallelism: enable SPMD before any device op, then build
    # the mesh. tp=1 leaves the single-chip path completely untouched.
    tp_mesh = None
    if tp and tp > 1:
        from . import sharding

        sharding.enable_spmd()
        tp_mesh = sharding.build_tp_mesh(tp)
        logger.info(f"[XTTS][TP] tp={tp}, mesh={tp_mesh}")

    # optimization_level 0 (default): the memory-layout optimizer probes ttnn op
    # constraints by allocating buffers on-device at compile time, which can
    # OOM when an earlier stage's weights are still resident. Levels >0 enable it
    # (better runtime, slower/heavier compile) -- try with care.
    torch_xla.set_custom_compile_options({"optimization_level": opt_level})

    # Persistent compilation cache -> warm subsequent runs. Must be initialized
    # before any XLA computation; guard so a re-entrant call does not assert.
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        if not torch_xla._XLAC._xla_computation_cache_is_initialized():
            xr.initialize_cache(cache_dir, readonly=False)
            logger.info(f"[XTTS] persistent compilation cache: {cache_dir}")
        else:
            logger.info("[XTTS] XLA computation cache already initialized; skipping")

    config = XTTSConfig(
        text=text,
        language=language,
        speaker_wav=speaker_wav,
        max_audio_tokens=max_audio_tokens,
        seed=seed,
    )
    pipeline = XTTSPipeline(config)
    pipeline.tp_mesh = tp_mesh
    setup_store = []
    with _timed(setup_store, "setup"):
        pipeline.setup()
    setup_secs = setup_store[0][1]

    wav = None
    for i in range(max(warmup, 0)):
        logger.info(f"[XTTS] warmup run {i + 1}/{warmup}")
        wav = pipeline.run()

    per_run = []  # list[dict[stage_label, seconds]], one per measured run
    for i in range(max(repeat, 0)):
        logger.info(f"[XTTS] measured run {i + 1}/{repeat}")
        wav = pipeline.run()
        per_run.append(dict(pipeline.timings))
        logger.info(
            f"[XTTS]   run total: {sum(s for _, s in pipeline.timings):.3f}s"
        )

    if wav is None:  # warmup == repeat == 0: still produce output
        wav = pipeline.run()

    save_wav(wav, output_path)
    logger.info(
        f"[XTTS] wrote {output_path} ({wav.shape[-1]} samples @ {OUTPUT_SAMPLE_RATE} Hz)"
    )
    _print_stats_table(setup_secs, per_run, warmup)
    return wav


if __name__ == "__main__":
    import argparse

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
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Un-timed warmup runs before measuring (pays one-time compile + "
        "on-device kernel-caching cost). Default: 1",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Measured runs; reports per-stage min/max/avg. Default: 3",
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="TT-MLIR optimization level (0=fastest compile/slowest runtime, "
        "2=slowest compile/fastest runtime). Levels >0 may OOM. Default: 0",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor-parallel degree. 1=single chip (default). >1 enables "
        "multi-chip Megatron TP on the GPT (needs device_count %% tp == 0, "
        "e.g. --tp 2 on an n300). See sharding.py / SHARDING_TP2.md.",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="Persistent PJRT compilation cache dir (pass '' to disable). Warm "
        "runs reuse compiled TT executables and skip recompilation. "
        f"Default: {DEFAULT_CACHE_DIR}",
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
        warmup=args.warmup,
        repeat=args.repeat,
        opt_level=args.opt_level,
        tp=args.tp,
        cache_dir=args.cache_dir,
    )
