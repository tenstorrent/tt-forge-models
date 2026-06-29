# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XTTS-v2 model loader implementation for text-to-speech tasks.

XTTS-v2 (coqui/XTTS-v2 on Hugging Face) is a multilingual voice-cloning TTS
model. The weights are hosted on Hugging Face and loaded through the maintained
``coqui-tts`` library (the original ``coqui-ai/TTS`` is archived). There is no
``transformers``-native path for this model.
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


class ModelVariant(StrEnum):
    """Available XTTS-v2 model variants."""

    XTTS_V2 = "XTTS_v2"


class ModelLoader(ForgeModel):
    """XTTS-v2 model loader implementation for text-to-speech tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.XTTS_V2: ModelConfig(
            pretrained_model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.XTTS_V2
    DEFAULT_TEXT = "Hello world, this is a test."
    DEFAULT_LANGUAGE = "en"
    # One of the 58 built-in studio speakers shipped in speakers_xtts.pth. Using a
    # built-in speaker provides precomputed, deterministic conditioning latents
    # without needing a reference audio file.
    DEFAULT_SPEAKER = "Claribel Dervla"
    # Length of the synthetic GPT audio-code sequence fed to the traced forward.
    GPT_CODE_LEN = 32

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

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

    def load_model(self, dtype_override=None):
        # XTTS-v2 weights are CPML-gated; downloading requires accepting the ToS.
        os.environ.setdefault("COQUI_TOS_AGREED", "1")

        # Applies the isin_mps_friendly shim and monkey patches Xtts.forward /
        # Xtts.post_process. Must run before any other TTS import.
        from .src import model  # noqa: F401

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.utils.manage import ModelManager

        model_name = self._variant_config.pretrained_model_name
        model_path, _, _ = ModelManager().download_model(model_name)

        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
        self.model.eval()

        # NOTE: dtype_override is intentionally ignored. XTTS's submodules do not
        # cast uniformly with ``.to(bfloat16)`` (some internal tensors stay
        # float32), which produces "mixed dtype (CPU)" layernorm errors in the
        # traced forward. The model is kept in float32 (as pocket_tts does); the
        # TT compiler handles device precision. The tester does not re-cast (its
        # dtype_override is None).

        return self.model

    def load_inputs(self, dtype_override=None, sample_text: Optional[str] = None):
        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        text = sample_text or self.DEFAULT_TEXT
        language = self.DEFAULT_LANGUAGE
        model_dtype = next(self.model.parameters()).dtype

        # Deterministic conditioning latents from a built-in studio speaker.
        speaker = self.model.speaker_manager.speakers[self.DEFAULT_SPEAKER]
        gpt_cond_latent = speaker["gpt_cond_latent"].to(model_dtype)
        speaker_embedding = speaker["speaker_embedding"].to(model_dtype)

        text_tokens = torch.IntTensor(
            self.model.tokenizer.encode(text.strip().lower(), lang=language)
        ).unsqueeze(0)
        text_len = torch.tensor([text_tokens.shape[-1]])

        # Fixed synthetic GPT audio codes instead of running the autoregressive
        # gpt.generate. The traced forward (GPT latents -> HiFiGAN decode) only
        # needs valid audio-token ids, so a deterministic fixed tensor keeps
        # load_inputs free of model inference while exercising the same graph.
        # Range excludes the start/stop tokens (the last two ids).
        #
        # Context: native Xtts.inference runs (1) gpt.generate -- the
        # autoregressive sampling loop that produces gpt_codes token-by-token
        # (incremental KV-cached GPT2 decode + mel_head logits + top_k/top_p/
        # temperature sampling), (2) a full-sequence gpt(..., return_latent=True)
        # over those codes -> latents, then (3) HiFiGAN decode. Stage (1) is a
        # dynamic-length, stochastic host loop, so it is neither statically
        # traceable nor reproducible for PCC. We therefore avoid stage (1)
        # entirely: the patched Xtts.forward (see src/model.py) keeps only the
        # deterministic traceable tail -- stages (2) + (3) -- and we feed these
        # fixed random gpt_codes in place of the sampled ones. This exercises the
        # identical heavy path (GPT2 trunk -> latents -> vocoder); only the
        # sampling loop and mel_head (the audio-token logit head used solely
        # inside generation) are not covered. Equivalence to native
        # Xtts.inference is PCC 1.0 on CPU.
        num_audio_tokens = self.model.args.gpt_num_audio_tokens
        generator = torch.Generator().manual_seed(0)
        gpt_codes = torch.randint(
            0,
            num_audio_tokens - 2,
            (1, self.GPT_CODE_LEN),
            generator=generator,
            dtype=torch.long,
        )
        expected_output_len = torch.tensor(
            [gpt_codes.shape[-1] * self.model.gpt.code_stride_len]
        )

        return {
            "text_tokens": text_tokens,
            "text_len": text_len,
            "gpt_codes": gpt_codes,
            "expected_output_len": expected_output_len,
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding,
        }

    def postprocess(self):
        """Return the decoded waveform produced during forward()."""
        return self.model.post_process()
