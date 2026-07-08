# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""XTTS-v2 (coqui/XTTS-v2) per-component model loaders for tt-xla bring-up.

XTTS-v2 is a multilingual voice-cloning TTS model loaded through ``coqui-tts``
(no ``transformers``-native path). It has no single traceable forward, so each
learned ``nn.Module`` is exposed as a separate variant -- a clean single-forward
graph that can be PCC-compared against CPU:

    SPEAKER_ENCODER  - ResNet-SE speaker encoder trunk (mel -> embedding)
    CONDITIONING     - GPT conditioning encoder + perceiver (mel -> cond latents)
    GPT_PREFILL      - GPT2 trunk + audio lm_head, prefill step (no KV cache)
    GPT_DECODE       - GPT2 trunk + audio lm_head, one KV-cached decode step
    GPT_LATENTS      - full-sequence GPT forward (codes -> GPT latents)
    HIFIGAN_DECODER  - HiFi-GAN vocoder (latents + speaker emb -> waveform)

Every component is fed the real tensors it would receive end-to-end, derived
from a real reference clip through the actual XTTS pipeline (deterministic
``gpt.generate`` so the audio codes are reproducible; no synthetic inputs).
"""

import os
from typing import Optional

import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

_PRETRAINED = "tts_models/multilingual/multi-dataset/xtts_v2"


class ModelVariant(StrEnum):
    """Available XTTS-v2 component variants (one traceable nn.Module each)."""

    SPEAKER_ENCODER = "speaker_encoder"
    CONDITIONING = "conditioning"
    GPT_PREFILL = "gpt_prefill"
    GPT_DECODE = "gpt_decode"
    GPT_LATENTS = "gpt_latents"
    HIFIGAN_DECODER = "hifigan_decoder"


class ModelLoader(ForgeModel):
    """Loads a single XTTS-v2 nn.Module component selected by ``variant``."""

    _VARIANTS = {
        ModelVariant.SPEAKER_ENCODER: ModelConfig(pretrained_model_name=_PRETRAINED),
        ModelVariant.CONDITIONING: ModelConfig(pretrained_model_name=_PRETRAINED),
        ModelVariant.GPT_PREFILL: ModelConfig(pretrained_model_name=_PRETRAINED),
        ModelVariant.GPT_DECODE: ModelConfig(pretrained_model_name=_PRETRAINED),
        ModelVariant.GPT_LATENTS: ModelConfig(pretrained_model_name=_PRETRAINED),
        ModelVariant.HIFIGAN_DECODER: ModelConfig(pretrained_model_name=_PRETRAINED),
    }

    DEFAULT_VARIANT = ModelVariant.GPT_LATENTS

    # Canonical example text/language from the coqui/XTTS-v2 Hugging Face model
    # card ("Direct model usage" / API examples).
    DEFAULT_TEXT = (
        "It took me quite a long time to develop a voice, and now that I have "
        "it I'm not going to be silent."
    )
    DEFAULT_LANGUAGE = "en"
    # Real public reference clip (LibriSpeech, the Whisper loader's asset), since
    # the model card ships only a placeholder speaker wav.
    REFERENCE_AUDIO = "test_files/pytorch/whisper/1272-128104-0000.pt"
    # Seconds of reference audio used for the conditioning latents (model-card
    # ``gpt_cond_len=3``).
    GPT_COND_LEN = 3
    XTTS_SR = 22050  # XTTS native sample rate
    SPEAKER_ENCODER_SR = 16000  # speaker encoder input rate (get_speaker_embedding)

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._xtts = None  # the full Xtts model (built once, shared by helpers)
        self._prefill_inputs = None  # cached GPT_PREFILL inputs (built with model)
        self._decode_state = None  # cached GPT_DECODE inputs + cache metadata
        self._cache = {}  # memoized real intermediate tensors (see _real helpers)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="xtts_v2",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    # ------------------------------------------------------------------ #
    # full-model build (shared across variants)                          #
    # ------------------------------------------------------------------ #
    def _build_xtts(self):
        if self._xtts is not None:
            return self._xtts

        # XTTS-v2 weights are CPML-gated; downloading requires accepting the ToS.
        os.environ.setdefault("COQUI_TOS_AGREED", "1")

        # Imports the isin_mps_friendly shim + component wrappers. Must run
        # before any other TTS import.
        from .src import model as _wrappers  # noqa: F401

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.utils.manage import ModelManager

        model_path, _, _ = ModelManager().download_model(
            self._variant_config.pretrained_model_name
        )
        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        xtts = Xtts.init_from_config(config)
        xtts.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
        xtts.eval()  # also runs gpt.init_gpt_for_inference() -> builds gpt_inference
        self._xtts = xtts
        return xtts

    # ------------------------------------------------------------------ #
    # real-input derivation (mirrors Xtts.get_conditioning_latents +      #
    # Xtts.inference). Each block is memoized so a variant only pays for   #
    # the tensors it needs; the whole chain is deterministic.             #
    # ------------------------------------------------------------------ #
    def _reference_audio_22k(self):
        """Load the real reference clip and resample to XTTS's 22.05 kHz rate."""
        if "audio22" in self._cache:
            return self._cache["audio22"]

        import numpy as np
        import torchaudio

        from ...tools.utils import get_file

        sample = torch.load(get_file(self.REFERENCE_AUDIO), weights_only=False)
        sr = int(sample["audio"].get("sampling_rate", 16000))  # asset's own rate
        audio = torch.tensor(
            np.asarray(sample["audio"]["array"], dtype="float32")
        ).unsqueeze(0)
        if sr != self.XTTS_SR:
            audio = torchaudio.functional.resample(audio, sr, self.XTTS_SR)
        self._cache["audio22"] = audio
        return audio

    def _text_tokens(self):
        if "text_tokens" not in self._cache:
            tokens = self._xtts.tokenizer.encode(
                self.DEFAULT_TEXT.strip().lower(), lang=self.DEFAULT_LANGUAGE
            )
            self._cache["text_tokens"] = torch.IntTensor(tokens).unsqueeze(0)
        return self._cache["text_tokens"]

    def _gpt_cond_latent(self):
        """Real GPT conditioning latents from the reference clip: (1, 32, 1024)."""
        if "gpt_cond_latent" not in self._cache:
            with torch.no_grad():
                self._cache["gpt_cond_latent"] = self._xtts.get_gpt_cond_latents(
                    self._reference_audio_22k(),
                    self.XTTS_SR,
                    length=self.GPT_COND_LEN,
                    chunk_length=self.GPT_COND_LEN,
                )
        return self._cache["gpt_cond_latent"]

    def _speaker_embedding(self):
        """Real speaker embedding from the reference clip: (1, 512, 1)."""
        if "speaker_embedding" not in self._cache:
            with torch.no_grad():
                self._cache["speaker_embedding"] = self._xtts.get_speaker_embedding(
                    self._reference_audio_22k(), self.XTTS_SR
                )
        return self._cache["speaker_embedding"]

    def _gpt_codes(self):
        """Real audio-code sequence from a deterministic (greedy) gpt.generate."""
        if "gpt_codes" not in self._cache:
            with torch.no_grad():
                self._cache["gpt_codes"] = self._xtts.gpt.generate(
                    cond_latents=self._gpt_cond_latent(),
                    text_inputs=self._text_tokens(),
                    do_sample=False,
                    num_beams=1,
                    num_return_sequences=1,
                    output_attentions=False,
                )
        return self._cache["gpt_codes"]

    def _gpt_latents(self):
        """Real GPT latents from the full-sequence forward: (1, L, 1024)."""
        if "gpt_latents" not in self._cache:
            gpt_codes = self._gpt_codes()
            text_tokens = self._text_tokens()
            expected_output_len = torch.tensor(
                [gpt_codes.shape[-1] * self._xtts.gpt.code_stride_len]
            )
            text_len = torch.tensor([text_tokens.shape[-1]])
            with torch.no_grad():
                self._cache["gpt_latents"] = self._xtts.gpt(
                    text_tokens,
                    text_len,
                    gpt_codes,
                    expected_output_len,
                    cond_latents=self._gpt_cond_latent(),
                    return_attentions=False,
                    return_latent=True,
                )
        return self._cache["gpt_latents"]

    def _build_decode_state(self):
        """Prefill ``[prefix, START]`` on CPU into a StaticCache and capture the
        inputs for one decode step (decoding the first generated token at audio
        position 1, as loop iteration 1 does).

        Returns ``(decode_inputs, valid_len, max_cache_len)``: ``valid_len`` is the
        slots the prefill wrote (the decode step's starting ``cumulative_length``);
        ``max_cache_len`` sizes the static buffers.
        """
        if self._decode_state is not None:
            return self._decode_state

        from transformers import StaticCache

        gpt = self._xtts.gpt
        gpt2 = gpt.gpt  # HF GPT2Model

        # Prefix embedding (cond latents + [START]text[STOP]); stored on
        # gpt_inference by compute_embeddings, same as the GPT_PREFILL path.
        gpt_cond_latent = self._gpt_cond_latent()
        text_tokens = self._text_tokens()
        with torch.no_grad():
            gpt.compute_embeddings(gpt_cond_latent, text_tokens)
        prefix_emb = gpt.gpt_inference.cached_prefix_emb.clone()  # [1, P, 1024]
        prefix_len = prefix_emb.shape[1]

        cfg = gpt2.config
        n_head = cfg.num_attention_heads
        head_dim = cfg.hidden_size // n_head
        # Prefill writes prefix_len + 1 slots ([prefix, START]); the decode step
        # writes one more, so the static cache needs prefix_len + 2 slots.
        max_cache_len = prefix_len + 2

        cache = StaticCache(config=cfg, max_cache_len=max_cache_len)
        cache.early_initialization(
            batch_size=1,
            num_heads=n_head,
            head_dim=head_dim,
            dtype=torch.float32,
            device="cpu",
        )

        def mask(valid):  # [1, max_cache_len]; 1s for written cache slots
            m = torch.zeros((1, max_cache_len), dtype=torch.long)
            m[:, :valid] = 1
            return m

        with torch.no_grad():
            # --- Prefill: [prefix, START(audio pos 0)] -> first audio token ---
            start_ids = torch.tensor([[gpt.start_audio_token]], dtype=torch.long)
            pos0 = torch.tensor([0], dtype=torch.long)
            emb = gpt.mel_embedding(start_ids)
            emb = emb + gpt.mel_pos_embedding.emb(pos0).unsqueeze(0)
            emb = torch.cat([prefix_emb.to(emb.dtype), emb], dim=1)
            out = gpt2(
                inputs_embeds=emb,
                attention_mask=mask(prefix_len + 1),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            logits = gpt.mel_head(gpt.final_norm(out.last_hidden_state))
            first_token = int(logits[:, -1, :].argmax(dim=-1).item())

        valid_len = prefix_len + 1  # cumulative_length written by the prefill
        cache_keys = torch.stack([layer.keys for layer in cache.layers])
        cache_values = torch.stack([layer.values for layer in cache.layers])

        decode_inputs = {
            "audio_ids": torch.tensor([[first_token]], dtype=torch.long),
            "positions": torch.tensor([1], dtype=torch.long),
            "attention_mask": mask(valid_len + 1),
            # cumulative_length the decode step starts from, as an input tensor
            # so the wrapper never creates a device-placed tensor inside forward.
            "cache_len": torch.tensor([valid_len], dtype=torch.long),
            "cache_keys": cache_keys,  # [n_layer, 1, n_head, max_cache_len, head_dim]
            "cache_values": cache_values,
        }
        self._decode_state = (decode_inputs, valid_len, max_cache_len)
        return self._decode_state

    # ------------------------------------------------------------------ #
    # NOTE: dtype_override is intentionally ignored. XTTS submodules do not
    # cast uniformly to bf16 (some internal tensors stay float32), which
    # produces "mixed dtype (CPU)" errors. The model is kept in float32; the TT
    # compiler handles device precision.
    # ------------------------------------------------------------------ #
    def load_model(self, dtype_override=None):
        from .src.model import (
            ConditioningWrapper,
            GptDecodeWrapper,
            GptLatentsWrapper,
            GptPrefillWrapper,
            HifiganDecoderWrapper,
            SpeakerEncoderWrapper,
        )

        xtts = self._build_xtts()

        if self._variant == ModelVariant.SPEAKER_ENCODER:
            return SpeakerEncoderWrapper(xtts)
        if self._variant == ModelVariant.CONDITIONING:
            return ConditioningWrapper(xtts)
        if self._variant == ModelVariant.GPT_LATENTS:
            return GptLatentsWrapper(xtts)
        if self._variant == ModelVariant.GPT_DECODE:
            _, valid_len, max_cache_len = self._build_decode_state()
            return GptDecodeWrapper(xtts, valid_len, max_cache_len)
        if self._variant == ModelVariant.HIFIGAN_DECODER:
            return HifiganDecoderWrapper(xtts)
        if self._variant == ModelVariant.GPT_PREFILL:
            # compute_embeddings builds the prefill token tensor and stores the
            # prefix embedding on gpt_inference; capture both deterministically.
            gpt_cond_latent = self._gpt_cond_latent()
            text_tokens = self._text_tokens()
            gpt_inputs = xtts.gpt.compute_embeddings(gpt_cond_latent, text_tokens)
            prefix_emb = xtts.gpt.gpt_inference.cached_prefix_emb.clone()
            attention_mask = torch.ones(
                1, gpt_inputs.shape[1], dtype=torch.bool
            )
            self._prefill_inputs = {
                "gpt_inputs": gpt_inputs,
                "attention_mask": attention_mask,
            }
            return GptPrefillWrapper(xtts, prefix_emb)

        raise ValueError(f"Unhandled variant: {self._variant}")

    def load_inputs(self, dtype_override=None):
        if self._xtts is None:
            self.load_model(dtype_override=dtype_override)

        if self._variant == ModelVariant.SPEAKER_ENCODER:
            # Speaker encoder consumes 16 kHz (get_speaker_embedding); the mel
            # front-end (torch.stft) is run on CPU here (complex FFT, #5216).
            import torchaudio

            audio_16k = torchaudio.functional.resample(
                self._reference_audio_22k(), self.XTTS_SR, self.SPEAKER_ENCODER_SR
            )
            with torch.no_grad():
                mel_spec = self._xtts.hifigan_decoder.speaker_encoder.torch_spec(
                    audio_16k
                )
            return {"mel_spec": mel_spec}

        if self._variant == ModelVariant.CONDITIONING:
            # Reference mel for the first GPT_COND_LEN s; mel params match
            # Xtts.get_gpt_cond_latents (perceiver path) exactly.
            from TTS.tts.models.xtts import wav_to_mel_cloning

            audio22 = self._reference_audio_22k()[:, : self.XTTS_SR * self.GPT_COND_LEN]
            cond_mel = wav_to_mel_cloning(
                audio22,
                mel_norms=self._xtts.mel_stats.cpu(),
                n_fft=2048,
                hop_length=256,
                win_length=1024,
                power=2,
                normalized=False,
                sample_rate=self.XTTS_SR,
                f_min=0,
                f_max=8000,
                n_mels=80,
            )
            return {"cond_mel": cond_mel}

        if self._variant == ModelVariant.GPT_PREFILL:
            if self._prefill_inputs is None:
                self.load_model(dtype_override=dtype_override)
            return self._prefill_inputs

        if self._variant == ModelVariant.GPT_DECODE:
            decode_inputs, _, _ = self._build_decode_state()
            return decode_inputs

        if self._variant == ModelVariant.GPT_LATENTS:
            gpt_codes = self._gpt_codes()
            text_tokens = self._text_tokens()
            text_len = torch.tensor([text_tokens.shape[-1]])
            expected_output_len = torch.tensor(
                [gpt_codes.shape[-1] * self._xtts.gpt.code_stride_len]
            )
            return {
                "text_tokens": text_tokens,
                "text_len": text_len,
                "gpt_codes": gpt_codes,
                "expected_output_len": expected_output_len,
                "gpt_cond_latent": self._gpt_cond_latent(),
            }

        if self._variant == ModelVariant.HIFIGAN_DECODER:
            return {
                "gpt_latents": self._gpt_latents(),
                "speaker_embedding": self._speaker_embedding(),
            }

        raise ValueError(f"Unhandled variant: {self._variant}")
