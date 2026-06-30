# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""XTTS-v2 (coqui/XTTS-v2) per-component model loaders for tt-xla bring-up.

XTTS-v2 is a multilingual voice-cloning TTS model whose weights live on Hugging
Face and are loaded through the maintained ``coqui-tts`` library (the original
``coqui-ai/TTS`` is archived). There is no ``transformers``-native path.

The model has no single traceable forward, so for Track-1 bring-up we expose its
individual ``nn.Module`` components as separate variants, each loadable as a
clean single-forward graph that can be PCC-compared against CPU:

    SPEAKER_ENCODER  - ResNet-SE speaker encoder (waveform -> embedding)
    CONDITIONING     - GPT conditioning encoder + perceiver (mel -> cond latents)
    GPT_PREFILL      - GPT2 trunk + audio lm_head, prefill step (no KV cache)
    GPT_LATENTS      - full-sequence GPT forward (codes -> GPT latents)
    HIFIGAN_DECODER  - HiFi-GAN vocoder (latents + speaker emb -> waveform)

The incremental KV-cached decode step uses the same GPT2 module as
GPT_PREFILL/GPT_LATENTS but is a stateful loop graph (static cache), so it is
brought up in Track 2 (the end-to-end pipeline), not as a standalone forward.
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
    GPT_LATENTS = "gpt_latents"
    HIFIGAN_DECODER = "hifigan_decoder"


class ModelLoader(ForgeModel):
    """Loads a single XTTS-v2 nn.Module component selected by ``variant``."""

    _VARIANTS = {
        ModelVariant.SPEAKER_ENCODER: ModelConfig(pretrained_model_name=_PRETRAINED),
        ModelVariant.CONDITIONING: ModelConfig(pretrained_model_name=_PRETRAINED),
        ModelVariant.GPT_PREFILL: ModelConfig(pretrained_model_name=_PRETRAINED),
        ModelVariant.GPT_LATENTS: ModelConfig(pretrained_model_name=_PRETRAINED),
        ModelVariant.HIFIGAN_DECODER: ModelConfig(pretrained_model_name=_PRETRAINED),
    }

    DEFAULT_VARIANT = ModelVariant.GPT_LATENTS

    DEFAULT_TEXT = "Hello world, this is a test."
    DEFAULT_LANGUAGE = "en"
    # One of the 58 built-in studio speakers in speakers_xtts.pth -> deterministic
    # precomputed conditioning latents without needing a reference audio file.
    DEFAULT_SPEAKER = "Claribel Dervla"
    # Length of the synthetic GPT audio-code sequence fed to GPT_LATENTS.
    GPT_CODE_LEN = 32
    # Reference-clip lengths for the encoder components (deterministic synthetic).
    SPEAKER_WAV_SECONDS = 3
    COND_MEL_FRAMES = 200
    # GPT latent dim / latent length used to drive the standalone vocoder.
    GPT_LATENT_DIM = 1024

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._xtts = None  # the full Xtts model (built once, shared by helpers)
        self._prefill_inputs = None  # cached GPT_PREFILL inputs (built with model)

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

    def _text_tokens(self):
        tokens = self._xtts.tokenizer.encode(
            self.DEFAULT_TEXT.strip().lower(), lang=self.DEFAULT_LANGUAGE
        )
        return torch.IntTensor(tokens).unsqueeze(0)

    def _speaker_latents(self):
        speaker = self._xtts.speaker_manager.speakers[self.DEFAULT_SPEAKER]
        return speaker["gpt_cond_latent"], speaker["speaker_embedding"]

    # ------------------------------------------------------------------ #
    # NOTE: dtype_override is intentionally ignored. XTTS submodules do not
    # cast uniformly to bf16 (some internal tensors stay float32), which
    # produces "mixed dtype (CPU)" errors. The model is kept in float32; the TT
    # compiler handles device precision.
    # ------------------------------------------------------------------ #
    def load_model(self, dtype_override=None):
        from .src.model import (
            ConditioningWrapper,
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
        if self._variant == ModelVariant.HIFIGAN_DECODER:
            return HifiganDecoderWrapper(xtts)
        if self._variant == ModelVariant.GPT_PREFILL:
            # compute_embeddings builds the prefill token tensor and stores the
            # prefix embedding on gpt_inference; capture both deterministically.
            gpt_cond_latent, _ = self._speaker_latents()
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

        gen = torch.Generator().manual_seed(0)

        if self._variant == ModelVariant.SPEAKER_ENCODER:
            audio_16k = 0.1 * torch.randn(
                1, 16000 * self.SPEAKER_WAV_SECONDS, generator=gen
            )
            return {"audio_16k": audio_16k}

        if self._variant == ModelVariant.CONDITIONING:
            cond_mel = torch.randn(1, 80, self.COND_MEL_FRAMES, generator=gen)
            return {"cond_mel": cond_mel}

        if self._variant == ModelVariant.GPT_PREFILL:
            if self._prefill_inputs is None:
                self.load_model(dtype_override=dtype_override)
            return self._prefill_inputs

        if self._variant == ModelVariant.GPT_LATENTS:
            gpt_cond_latent, _ = self._speaker_latents()
            text_tokens = self._text_tokens()
            text_len = torch.tensor([text_tokens.shape[-1]])
            # Fixed audio codes in place of the sampled ones (range excludes the
            # start/stop tokens, the last two ids).
            num_audio_tokens = self._xtts.args.gpt_num_audio_tokens
            gpt_codes = torch.randint(
                0,
                num_audio_tokens - 2,
                (1, self.GPT_CODE_LEN),
                generator=gen,
                dtype=torch.long,
            )
            expected_output_len = torch.tensor(
                [gpt_codes.shape[-1] * self._xtts.gpt.code_stride_len]
            )
            return {
                "text_tokens": text_tokens,
                "text_len": text_len,
                "gpt_codes": gpt_codes,
                "expected_output_len": expected_output_len,
                "gpt_cond_latent": gpt_cond_latent,
            }

        if self._variant == ModelVariant.HIFIGAN_DECODER:
            _, speaker_embedding = self._speaker_latents()
            gpt_latents = 0.1 * torch.randn(
                1, self.GPT_CODE_LEN, self.GPT_LATENT_DIM, generator=gen
            )
            return {
                "gpt_latents": gpt_latents,
                "speaker_embedding": speaker_embedding,
            }

        raise ValueError(f"Unhandled variant: {self._variant}")
